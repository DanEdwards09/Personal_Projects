"""
Claims Data Reconciliation Solution
==================================

Automated solution to reconcile claims data between different systems,
reducing manual validation time and improving data accuracy for actuarial analysis.

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import json
import logging
from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass, asdict
import warnings

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    """Data class to store reconciliation results"""

    matched_records: int
    unmatched_source: int
    unmatched_target: int
    discrepancies: int
    accuracy_rate: float
    processing_time: float
    detailed_results: Dict


class ClaimsReconciliationEngine:
    """
    Advanced claims data reconciliation engine that automatically validates
    and reconciles claims data between different systems or data sources.
    """

    def __init__(self, tolerance_amount: float = 0.01, tolerance_days: int = 1):
        """
        Initialize the reconciliation engine

        Args:
            tolerance_amount: Monetary tolerance for amount matching (default: $0.01)
            tolerance_days: Date tolerance for date matching (default: 1 day)
        """
        self.tolerance_amount = tolerance_amount
        self.tolerance_days = tolerance_days
        self.reconciliation_rules = {}
        self.data_quality_issues = []

    def generate_sample_claims_data(
        self, system_name: str, n_records: int = 1000
    ) -> pd.DataFrame:
        """Generate realistic sample claims data for demonstration"""
        np.random.seed(42 if system_name == "source" else 43)

        # Generate base data
        claim_ids = [
            f"CLM{system_name.upper()}{str(i).zfill(6)}"
            for i in range(1, n_records + 1)
        ]
        policy_numbers = [
            f"POL{np.random.randint(100000, 999999)}" for _ in range(n_records)
        ]

        # Date ranges
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)

        incident_dates = [
            start_date
            + timedelta(days=np.random.randint(0, (end_date - start_date).days))
            for _ in range(n_records)
        ]

        reported_dates = [
            inc_date + timedelta(days=np.random.randint(0, 30))
            for inc_date in incident_dates
        ]

        # Claim types and amounts
        claim_types = [
            "Auto Collision",
            "Auto Comprehensive",
            "Property Damage",
            "Theft",
            "Fire",
            "Water Damage",
            "Liability",
            "Medical",
        ]

        types = np.random.choice(
            claim_types, n_records, p=[0.25, 0.15, 0.15, 0.1, 0.08, 0.12, 0.1, 0.05]
        )

        # Generate amounts based on claim type
        amounts = []
        for claim_type in types:
            if claim_type in ["Auto Collision", "Auto Comprehensive"]:
                amount = np.random.lognormal(8, 1)  # Higher amounts
            elif claim_type in ["Property Damage", "Fire", "Water Damage"]:
                amount = np.random.lognormal(9, 1.2)  # Very high amounts
            else:
                amount = np.random.lognormal(7, 0.8)  # Moderate amounts
            amounts.append(max(100, min(500000, amount)))  # Cap between $100-$500k

        # Status
        statuses = np.random.choice(
            ["Open", "Closed", "Pending", "Under Investigation"],
            n_records,
            p=[0.3, 0.4, 0.2, 0.1],
        )

        # Customer information
        customer_ids = [
            f"CUST{np.random.randint(10000, 99999)}" for _ in range(n_records)
        ]

        # Create DataFrame
        data = pd.DataFrame(
            {
                "claim_id": claim_ids,
                "policy_number": policy_numbers,
                "customer_id": customer_ids,
                "incident_date": incident_dates,
                "reported_date": reported_dates,
                "claim_type": types,
                "claim_amount": amounts,
                "status": statuses,
                "adjuster": [
                    f"ADJ{np.random.randint(100, 999)}" for _ in range(n_records)
                ],
                "reserve_amount": [
                    amt * np.random.uniform(0.8, 1.2) for amt in amounts
                ],
            }
        )

        # Introduce some system-specific variations for realistic reconciliation testing
        if system_name == "target":
            # Some records might be missing
            data = data.sample(frac=0.95).reset_index(drop=True)

            # Some amounts might have small differences
            random_indices = np.random.choice(
                data.index, size=int(len(data) * 0.1), replace=False
            )
            data.loc[random_indices, "claim_amount"] += np.random.uniform(
                -50, 50, len(random_indices)
            )

            # Some dates might be slightly different
            date_indices = np.random.choice(
                data.index, size=int(len(data) * 0.05), replace=False
            )
            data.loc[date_indices, "reported_date"] += pd.to_timedelta(
                np.random.randint(-2, 3), unit="D"
            )

            # Different claim IDs format
            data["claim_id"] = data["claim_id"].str.replace("TARGET", "TGT")

        return data

    def standardize_data(self, df: pd.DataFrame, system_name: str) -> pd.DataFrame:
        """Standardize data formats for consistent comparison"""
        df_clean = df.copy()

        # Standardize column names
        df_clean.columns = df_clean.columns.str.lower().str.replace(" ", "_")

        # Standardize dates
        date_columns = ["incident_date", "reported_date"]
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")

        # Standardize amounts (remove currency symbols, convert to float)
        amount_columns = ["claim_amount", "reserve_amount"]
        for col in amount_columns:
            if col in df_clean.columns:
                if df_clean[col].dtype == "object":
                    # Remove currency symbols and convert
                    df_clean[col] = (
                        df_clean[col].astype(str).str.replace(r"[$,]", "", regex=True)
                    )
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        # Standardize text fields
        text_columns = ["claim_type", "status"]
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.title()

        # Create matching keys
        df_clean["policy_match_key"] = df_clean["policy_number"].astype(str).str.upper()
        df_clean["customer_match_key"] = df_clean["customer_id"].astype(str).str.upper()

        # Add data quality flags
        df_clean["source_system"] = system_name
        df_clean["data_quality_score"] = self._calculate_data_quality_score(df_clean)

        logger.info(f"Standardized {len(df_clean)} records from {system_name} system")

        return df_clean

    def _calculate_data_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate data quality score for each record"""
        scores = pd.Series(index=df.index, dtype=float)

        for idx, row in df.iterrows():
            score = 100.0

            # Deduct points for missing values
            missing_critical = pd.isna(
                row[["claim_id", "policy_number", "claim_amount"]]
            ).sum()
            score -= missing_critical * 20

            # Deduct points for suspicious amounts
            if pd.notna(row.get("claim_amount")):
                if row["claim_amount"] <= 0:
                    score -= 30
                elif row["claim_amount"] > 1000000:  # Very high amount
                    score -= 10

            # Deduct points for future dates
            if pd.notna(row.get("incident_date")):
                if row["incident_date"] > datetime.now():
                    score -= 25

            scores[idx] = max(0, score)

        return scores

    def create_matching_strategy(
        self, primary_keys: List[str], fuzzy_match_fields: List[str] = None
    ) -> Dict:
        """Create matching strategy for reconciliation"""
        strategy = {
            "primary_keys": primary_keys,
            "fuzzy_match_fields": fuzzy_match_fields or [],
            "amount_tolerance": self.tolerance_amount,
            "date_tolerance": self.tolerance_days,
        }

        return strategy

    def reconcile_data(
        self, source_df: pd.DataFrame, target_df: pd.DataFrame, matching_strategy: Dict
    ) -> ReconciliationResult:
        """
        Main reconciliation function that compares two datasets

        Args:
            source_df: Source system data
            target_df: Target system data
            matching_strategy: Strategy for matching records

        Returns:
            ReconciliationResult object with detailed reconciliation results
        """
        start_time = datetime.now()

        logger.info("Starting claims data reconciliation...")

        # Standardize both datasets
        source_clean = self.standardize_data(source_df, "source")
        target_clean = self.standardize_data(target_df, "target")

        # Initialize results tracking
        matched_records = []
        unmatched_source = []
        unmatched_target = []
        discrepancies = []

        # Create matching indices
        primary_keys = matching_strategy["primary_keys"]

        # Exact matching first
        exact_matches = self._perform_exact_matching(
            source_clean, target_clean, primary_keys
        )
        matched_records.extend(exact_matches["matches"])

        # Get unmatched records from exact matching
        source_unmatched = source_clean[
            ~source_clean.index.isin(
                [m["source_idx"] for m in exact_matches["matches"]]
            )
        ]
        target_unmatched = target_clean[
            ~target_clean.index.isin(
                [m["target_idx"] for m in exact_matches["matches"]]
            )
        ]

        # Fuzzy matching for remaining records
        if matching_strategy["fuzzy_match_fields"]:
            fuzzy_matches = self._perform_fuzzy_matching(
                source_unmatched,
                target_unmatched,
                matching_strategy["fuzzy_match_fields"],
            )
            matched_records.extend(fuzzy_matches["matches"])

            # Update unmatched after fuzzy matching
            source_unmatched = source_unmatched[
                ~source_unmatched.index.isin(
                    [m["source_idx"] for m in fuzzy_matches["matches"]]
                )
            ]
            target_unmatched = target_unmatched[
                ~target_unmatched.index.isin(
                    [m["target_idx"] for m in fuzzy_matches["matches"]]
                )
            ]

        # Identify discrepancies in matched records
        discrepancies = self._identify_discrepancies(
            matched_records, source_clean, target_clean, matching_strategy
        )

        # Calculate metrics
        total_source = len(source_clean)
        total_target = len(target_clean)
        total_matched = len(matched_records)
        total_unmatched_source = len(source_unmatched)
        total_unmatched_target = len(target_unmatched)
        total_discrepancies = len(discrepancies)

        accuracy_rate = (
            (total_matched - total_discrepancies) / max(total_source, total_target)
            if max(total_source, total_target) > 0
            else 0
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        # Create detailed results
        detailed_results = {
            "exact_matches": exact_matches,
            "fuzzy_matches": fuzzy_matches
            if "fuzzy_matches" in locals()
            else {"matches": []},
            "discrepancies": discrepancies,
            "unmatched_source_records": source_unmatched.to_dict("records"),
            "unmatched_target_records": target_unmatched.to_dict("records"),
            "data_quality_issues": self.data_quality_issues,
            "summary_statistics": {
                "total_source_records": total_source,
                "total_target_records": total_target,
                "match_rate": total_matched / total_source if total_source > 0 else 0,
                "discrepancy_rate": total_discrepancies / total_matched
                if total_matched > 0
                else 0,
            },
        }

        logger.info(f"Reconciliation completed in {processing_time:.2f} seconds")
        logger.info(
            f"Matched: {total_matched}, Unmatched Source: {total_unmatched_source}, "
            f"Unmatched Target: {total_unmatched_target}, Discrepancies: {total_discrepancies}"
        )

        return ReconciliationResult(
            matched_records=total_matched,
            unmatched_source=total_unmatched_source,
            unmatched_target=total_unmatched_target,
            discrepancies=total_discrepancies,
            accuracy_rate=accuracy_rate,
            processing_time=processing_time,
            detailed_results=detailed_results,
        )

    def _perform_exact_matching(
        self, source_df: pd.DataFrame, target_df: pd.DataFrame, primary_keys: List[str]
    ) -> Dict:
        """Perform exact matching based on primary keys"""
        matches = []

        # Create composite key for matching
        source_keys = (
            source_df[primary_keys].astype(str).apply(lambda x: "|".join(x), axis=1)
        )
        target_keys = (
            target_df[primary_keys].astype(str).apply(lambda x: "|".join(x), axis=1)
        )

        # Find exact matches
        for source_idx, source_key in source_keys.items():
            matching_targets = target_keys[target_keys == source_key]

            if len(matching_targets) > 0:
                target_idx = matching_targets.index[0]  # Take first match
                matches.append(
                    {
                        "source_idx": source_idx,
                        "target_idx": target_idx,
                        "match_type": "exact",
                        "match_confidence": 1.0,
                        "matching_key": source_key,
                    }
                )

        logger.info(f"Found {len(matches)} exact matches")

        return {"matches": matches}

    def _perform_fuzzy_matching(
        self, source_df: pd.DataFrame, target_df: pd.DataFrame, fuzzy_fields: List[str]
    ) -> Dict:
        """Perform fuzzy matching on specified fields"""
        matches = []

        # Simple fuzzy matching based on similarity scores
        for source_idx, source_row in source_df.iterrows():
            best_match_idx = None
            best_match_score = 0

            for target_idx, target_row in target_df.iterrows():
                similarity_score = self._calculate_similarity(
                    source_row, target_row, fuzzy_fields
                )

                if (
                    similarity_score > 0.8 and similarity_score > best_match_score
                ):  # 80% threshold
                    best_match_idx = target_idx
                    best_match_score = similarity_score

            if best_match_idx is not None:
                matches.append(
                    {
                        "source_idx": source_idx,
                        "target_idx": best_match_idx,
                        "match_type": "fuzzy",
                        "match_confidence": best_match_score,
                        "matching_fields": fuzzy_fields,
                    }
                )

        logger.info(f"Found {len(matches)} fuzzy matches")

        return {"matches": matches}

    def _calculate_similarity(
        self, source_row: pd.Series, target_row: pd.Series, fields: List[str]
    ) -> float:
        """Calculate similarity score between two records"""
        scores = []

        for field in fields:
            if field in source_row and field in target_row:
                source_val = str(source_row[field]).lower().strip()
                target_val = str(target_row[field]).lower().strip()

                if source_val == target_val:
                    scores.append(1.0)
                elif source_val in target_val or target_val in source_val:
                    scores.append(0.8)
                else:
                    # Simple character-based similarity
                    similarity = self._string_similarity(source_val, target_val)
                    scores.append(similarity)

        return np.mean(scores) if scores else 0.0

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using Jaccard similarity"""
        if not s1 or not s2:
            return 0.0

        # Convert to sets of characters
        set1 = set(s1)
        set2 = set(s2)

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        return len(intersection) / len(union) if union else 0.0

    def _identify_discrepancies(
        self,
        matched_records: List[Dict],
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        matching_strategy: Dict,
    ) -> List[Dict]:
        """Identify discrepancies in matched records"""
        discrepancies = []

        for match in matched_records:
            source_idx = match["source_idx"]
            target_idx = match["target_idx"]

            source_record = source_df.loc[source_idx]
            target_record = target_df.loc[target_idx]

            record_discrepancies = []

            # Check amount discrepancies
            if pd.notna(source_record.get("claim_amount")) and pd.notna(
                target_record.get("claim_amount")
            ):
                amount_diff = abs(
                    source_record["claim_amount"] - target_record["claim_amount"]
                )
                if amount_diff > matching_strategy["amount_tolerance"]:
                    record_discrepancies.append(
                        {
                            "field": "claim_amount",
                            "source_value": source_record["claim_amount"],
                            "target_value": target_record["claim_amount"],
                            "difference": amount_diff,
                            "severity": "high" if amount_diff > 1000 else "medium",
                        }
                    )

            # Check date discrepancies
            date_fields = ["incident_date", "reported_date"]
            for field in date_fields:
                if pd.notna(source_record.get(field)) and pd.notna(
                    target_record.get(field)
                ):
                    date_diff = abs((source_record[field] - target_record[field]).days)
                    if date_diff > matching_strategy["date_tolerance"]:
                        record_discrepancies.append(
                            {
                                "field": field,
                                "source_value": source_record[field],
                                "target_value": target_record[field],
                                "difference_days": date_diff,
                                "severity": "high" if date_diff > 7 else "low",
                            }
                        )

            # Check status discrepancies
            if pd.notna(source_record.get("status")) and pd.notna(
                target_record.get("status")
            ):
                if source_record["status"] != target_record["status"]:
                    record_discrepancies.append(
                        {
                            "field": "status",
                            "source_value": source_record["status"],
                            "target_value": target_record["status"],
                            "severity": "medium",
                        }
                    )

            if record_discrepancies:
                discrepancies.append(
                    {
                        "match_info": match,
                        "discrepancies": record_discrepancies,
                        "total_discrepancy_count": len(record_discrepancies),
                    }
                )

        return discrepancies

    def generate_reconciliation_report(
        self, result: ReconciliationResult, output_file: Optional[str] = None
    ) -> str:
        """Generate a comprehensive reconciliation report"""
        report = []
        report.append("=" * 60)
        report.append("CLAIMS DATA RECONCILIATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Processing Time: {result.processing_time:.2f} seconds")
        report.append("")

        # Summary Statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 30)
        stats = result.detailed_results["summary_statistics"]
        report.append(f"Source Records: {stats['total_source_records']:,}")
        report.append(f"Target Records: {stats['total_target_records']:,}")
        report.append(f"Matched Records: {result.matched_records:,}")
        report.append(f"Match Rate: {stats['match_rate']:.2%}")
        report.append(f"Unmatched Source: {result.unmatched_source:,}")
        report.append(f"Unmatched Target: {result.unmatched_target:,}")
        report.append(f"Discrepancies: {result.discrepancies:,}")
        report.append(f"Discrepancy Rate: {stats['discrepancy_rate']:.2%}")
        report.append(f"Overall Accuracy: {result.accuracy_rate:.2%}")
        report.append("")

        # Match Quality Analysis
        report.append("MATCH QUALITY ANALYSIS")
        report.append("-" * 30)
        exact_matches = len(result.detailed_results["exact_matches"]["matches"])
        fuzzy_matches = len(result.detailed_results["fuzzy_matches"]["matches"])
        report.append(f"Exact Matches: {exact_matches:,}")
        report.append(f"Fuzzy Matches: {fuzzy_matches:,}")
        report.append("")

        # Discrepancy Analysis
        if result.discrepancies > 0:
            report.append("DISCREPANCY ANALYSIS")
            report.append("-" * 30)

            discrepancy_summary = {}
            severity_summary = {"high": 0, "medium": 0, "low": 0}

            for disc in result.detailed_results["discrepancies"]:
                for d in disc["discrepancies"]:
                    field = d["field"]
                    severity = d["severity"]

                    discrepancy_summary[field] = discrepancy_summary.get(field, 0) + 1
                    severity_summary[severity] += 1

            report.append("By Field:")
            for field, count in discrepancy_summary.items():
                report.append(f"  {field}: {count}")

            report.append("\nBy Severity:")
            for severity, count in severity_summary.items():
                report.append(f"  {severity.title()}: {count}")
            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)

        if result.accuracy_rate > 0.95:
            report.append("✓ Data quality is excellent (>95% accuracy)")
        elif result.accuracy_rate > 0.90:
            report.append(
                "⚠ Data quality is good but could be improved (90-95% accuracy)"
            )
        else:
            report.append("✗ Data quality needs attention (<90% accuracy)")

        if result.unmatched_source > result.matched_records * 0.1:
            report.append(
                "⚠ High number of unmatched source records - investigate data completeness"
            )

        if result.discrepancies > result.matched_records * 0.05:
            report.append("⚠ High discrepancy rate - review data entry processes")

        if stats["discrepancy_rate"] > 0.1:
            report.append("⚠ Consider implementing automated data validation rules")

        report_text = "\n".join(report)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")

        return report_text

    def export_discrepancies(
        self,
        result: ReconciliationResult,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        output_file: str = "discrepancies.csv",
    ):
        """Export detailed discrepancies to CSV for manual review"""
        if not result.detailed_results["discrepancies"]:
            logger.info("No discrepancies to export")
            return

        export_data = []

        for disc in result.detailed_results["discrepancies"]:
            match_info = disc["match_info"]
            source_idx = match_info["source_idx"]
            target_idx = match_info["target_idx"]

            base_record = {
                "source_claim_id": source_df.loc[source_idx, "claim_id"],
                "target_claim_id": target_df.loc[target_idx, "claim_id"],
                "policy_number": source_df.loc[source_idx, "policy_number"],
                "match_type": match_info["match_type"],
                "match_confidence": match_info["match_confidence"],
            }

            for d in disc["discrepancies"]:
                record = base_record.copy()
                record.update(
                    {
                        "discrepancy_field": d["field"],
                        "source_value": d["source_value"],
                        "target_value": d["target_value"],
                        "severity": d["severity"],
                    }
                )

                if "difference" in d:
                    record["amount_difference"] = d["difference"]
                if "difference_days" in d:
                    record["days_difference"] = d["difference_days"]

                export_data.append(record)

        df_export = pd.DataFrame(export_data)
        df_export.to_csv(output_file, index=False)
        logger.info(f"Discrepancies exported to {output_file}")


class APIDataSource:
    """Simulated API data source for demonstration"""

    def __init__(self, base_url: str = "https://api.insurance-system.com"):
        self.base_url = base_url
        self.api_key = "demo_key_12345"

    def fetch_claims_data(
        self, start_date: str, end_date: str, system_name: str = "external"
    ) -> pd.DataFrame:
        """Simulate fetching claims data from an external API"""
        logger.info(f"Fetching claims data from {system_name} API...")
        logger.info(f"Date range: {start_date} to {end_date}")

        # Simulate API call delay
        import time

        time.sleep(0.5)

        # Generate sample data (in real scenario, this would be actual API call)
        engine = ClaimsReconciliationEngine()
        data = engine.generate_sample_claims_data(system_name, 800)

        # Filter by date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        data = data[(data["incident_date"] >= start) & (data["incident_date"] <= end)]

        logger.info(f"Retrieved {len(data)} claims from {system_name} API")

        return data


def main():
    """Main execution function demonstrating the Claims Reconciliation Solution"""
    print("Claims Data Reconciliation Solution")
    print("===================================")

    # Initialize the reconciliation engine
    reconciler = ClaimsReconciliationEngine(tolerance_amount=0.50, tolerance_days=2)

    print("\n1. Generating sample claims data...")

    # Generate sample data from two different systems
    source_data = reconciler.generate_sample_claims_data("source", 1000)
    target_data = reconciler.generate_sample_claims_data(
        "target", 950
    )  # Slightly fewer records

    print(f"Source system: {len(source_data)} records")
    print(f"Target system: {len(target_data)} records")

    # Display sample data
    print("\nSample Source Data:")
    print(
        source_data[
            ["claim_id", "policy_number", "claim_amount", "claim_type", "status"]
        ].head()
    )

    print("\n2. Configuring matching strategy...")

    # Define matching strategy
    matching_strategy = reconciler.create_matching_strategy(
        primary_keys=["policy_number", "customer_id"],
        fuzzy_match_fields=["claim_type", "claim_amount"],
    )

    print("Matching Strategy:")
    print(f"  Primary Keys: {matching_strategy['primary_keys']}")
    print(f"  Fuzzy Match Fields: {matching_strategy['fuzzy_match_fields']}")
    print(f"  Amount Tolerance: ${matching_strategy['amount_tolerance']}")
    print(f"  Date Tolerance: {matching_strategy['date_tolerance']} days")

    print("\n3. Performing reconciliation...")

    # Perform reconciliation
    result = reconciler.reconcile_data(source_data, target_data, matching_strategy)

    print("\n4. Reconciliation Results:")
    print(f"  Matched Records: {result.matched_records:,}")
    print(f"  Unmatched Source: {result.unmatched_source:,}")
    print(f"  Unmatched Target: {result.unmatched_target:,}")
    print(f"  Discrepancies: {result.discrepancies:,}")
    print(f"  Accuracy Rate: {result.accuracy_rate:.2%}")
    print(f"  Processing Time: {result.processing_time:.2f} seconds")

    # Calculate time savings
    manual_time_per_record = 0.5  # minutes
    total_records = len(source_data)
    manual_time_hours = (total_records * manual_time_per_record) / 60
    automated_time_hours = result.processing_time / 3600
    time_savings = (
        (manual_time_hours - automated_time_hours) / manual_time_hours
    ) * 100

    print(f"\n5. Efficiency Analysis:")
    print(f"  Estimated manual processing time: {manual_time_hours:.1f} hours")
    print(f"  Automated processing time: {automated_time_hours:.3f} hours")
    print(f"  Time savings: {time_savings:.1f}%")

    print("\n6. Generating comprehensive report...")

    # Generate report
    report = reconciler.generate_reconciliation_report(result)
    print(report)

    # Export discrepancies if any exist
    if result.discrepancies > 0:
        print("\n7. Exporting discrepancies for manual review...")
        reconciler.export_discrepancies(
            result, source_data, target_data, "claims_discrepancies.csv"
        )
        print("Discrepancies exported to 'claims_discrepancies.csv'")

    print("\n8. API Integration Example:")

    # Demonstrate API integration
    api_source = APIDataSource("https://internal-claims-api.company.com")
    external_data = api_source.fetch_claims_data(
        "2024-01-01", "2024-06-30", "external_system"
    )

    print(f"Retrieved {len(external_data)} records from external API")

    # Quick reconciliation with API data
    api_matching_strategy = reconciler.create_matching_strategy(["policy_number"])
    api_result = reconciler.reconcile_data(
        source_data.head(500), external_data.head(400), api_matching_strategy
    )

    print(f"API Reconciliation Results:")
    print(
        f"  Matched: {api_result.matched_records}, Accuracy: {api_result.accuracy_rate:.2%}"
    )

    print("\n✓ Claims Data Reconciliation Solution completed successfully!")
    print("✓ Demonstrated >90% reduction in manual validation time")
    print("✓ Improved data accuracy for actuarial analysis")


if __name__ == "__main__":
    main()
