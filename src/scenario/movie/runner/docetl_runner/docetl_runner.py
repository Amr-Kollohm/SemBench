"""
DocETL system runner implementation for movie scenario.
"""

import json
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_docetl_runner.generic_docetl_runner import (
    GenericDocETLRunner,
)


class DocETLRunner(GenericDocETLRunner):
    """Runner for DocETL system (movie scenario)."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gpt-4o-mini",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )

    def _execute_q1(self) -> dict:
        """
        Execute Q1: Find five clearly positive movie reviews.

        Returns:
            Dict with keys: "results" and "stats"
        """
        try:
            from docetl.api import (
                Dataset,
                FilterOp,
                Pipeline,
                PipelineOutput,
                PipelineStep,
            )
        except ImportError as e:
            raise RuntimeError(
                "DocETL is not installed in this environment. "
                "Install with: uv pip install docetl 'pyrate-limiter<4'"
            ) from e

        reviews_path = (self.data_path / "Reviews.csv").resolve()
        output_path = (self.results_path / "_docetl_q1_output.json").resolve()
        intermediate_dir = (
            self.results_path / "_docetl_q1_intermediate"
        ).resolve()

        dataset = Dataset(
            type="file",
            path=str(reviews_path),
            source="local",
        )

        filter_op = FilterOp(
            name="filter_positive_reviews",
            type="filter",
            limit=5,
            prompt=(
                "Determine if the following movie review is clearly positive.\n\n"
                "Review: {{ input.reviewText }}"
            ),
            output={"schema": {"is_positive": "bool"}},
        )

        step = PipelineStep(
            name="filter_positive",
            input="reviews",
            operations=["filter_positive_reviews"],
        )

        output = PipelineOutput(
            type="file",
            path=str(output_path),
            intermediate_dir=str(intermediate_dir),
        )

        pipeline = Pipeline(
            name="q1_positive_reviews",
            datasets={"reviews": dataset},
            operations=[filter_op],
            steps=[step],
            output=output,
            default_model=self.model_name,
        )

        cost = pipeline.run(max_threads=self.concurrent_llm_worker)
        stats = {"cost": cost, "token_usage": {}}

        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        results_df = pd.DataFrame(payload) if isinstance(payload, list) else pd.DataFrame()

        if "reviewId" in results_df.columns:
            results_df = results_df[["reviewId"]].head(5)
        else:
            results_df = pd.DataFrame(columns=["reviewId"])

        if len(results_df) == 0:
            results_df = pd.DataFrame(columns=["reviewId"])

        return {"results": results_df, "stats": stats}

    def _execute_q2(self) -> dict:
        """
        Execute Q2: Find five positive reviews for movie "taken_3".

        Returns:
            Dict with keys: "results" and "stats"
        """
        try:
            from docetl.api import (
                CodeFilterOp,
                Dataset,
                FilterOp,
                Pipeline,
                PipelineOutput,
                PipelineStep,
            )
        except ImportError as e:
            raise RuntimeError(
                "DocETL is not installed in this environment. "
                "Install with: uv pip install docetl 'pyrate-limiter<4'"
            ) from e

        reviews_path = (self.data_path / "Reviews.csv").resolve()
        output_path = (self.results_path / "_docetl_q2_output.json").resolve()
        intermediate_dir = (
            self.results_path / "_docetl_q2_intermediate"
        ).resolve()

        dataset = Dataset(
            type="file",
            path=str(reviews_path),
            source="local",
        )

        movie_filter = CodeFilterOp(
            name="filter_by_movie",
            type="code_filter",
            code=(
                "def transform(doc):\n"
                "    return doc.get('id') == 'taken_3'"
            ),
        )

        positive_filter = FilterOp(
            name="filter_positive_reviews",
            type="filter",
            limit=5,
            prompt=(
                "Determine if the following movie review is clearly positive.\n\n"
                "Review: {{ input.reviewText }}"
            ),
            output={"schema": {"is_positive": "bool"}},
        )

        step = PipelineStep(
            name="find_positive_reviews",
            input="reviews",
            operations=["filter_by_movie", "filter_positive_reviews"],
        )

        output = PipelineOutput(
            type="file",
            path=str(output_path),
            intermediate_dir=str(intermediate_dir),
        )

        pipeline = Pipeline(
            name="q2_positive_reviews_taken_3",
            datasets={"reviews": dataset},
            operations=[movie_filter, positive_filter],
            steps=[step],
            output=output,
            default_model=self.model_name,
        )

        cost = pipeline.run(max_threads=self.concurrent_llm_worker)
        stats = {"cost": cost, "token_usage": {}}

        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        results_df = (
            pd.DataFrame(payload)
            if isinstance(payload, list)
            else pd.DataFrame()
        )

        if "reviewId" in results_df.columns:
            results_df = results_df[["reviewId"]].head(5)
        else:
            results_df = pd.DataFrame(columns=["reviewId"])

        if len(results_df) == 0:
            results_df = pd.DataFrame(columns=["reviewId"])

        return {"results": results_df, "stats": stats}

    def _execute_q3(self) -> dict:
        """
        Execute Q3: Count positive reviews for movie "taken_3".

        Returns:
            Dict with keys: "results" and "stats"
        """
        try:
            from docetl.api import (
                CodeFilterOp,
                Dataset,
                FilterOp,
                Pipeline,
                PipelineOutput,
                PipelineStep,
            )
        except ImportError as e:
            raise RuntimeError(
                "DocETL is not installed in this environment. "
                "Install with: uv pip install docetl 'pyrate-limiter<4'"
            ) from e

        reviews_path = (self.data_path / "Reviews.csv").resolve()
        output_path = (self.results_path / "_docetl_q3_output.json").resolve()
        intermediate_dir = (
            self.results_path / "_docetl_q3_intermediate"
        ).resolve()

        dataset = Dataset(
            type="file",
            path=str(reviews_path),
            source="local",
        )

        movie_filter = CodeFilterOp(
            name="filter_by_movie",
            type="code_filter",
            code=(
                "def transform(doc):\n"
                "    return doc.get('id') == 'taken_3'"
            ),
        )

        positive_filter = FilterOp(
            name="filter_positive_reviews",
            type="filter",
            prompt=(
                "Determine if the following movie review is clearly positive.\n\n"
                "Review: {{ input.reviewText }}"
            ),
            output={"schema": {"is_positive": "bool"}},
        )

        step = PipelineStep(
            name="count_positive_reviews",
            input="reviews",
            operations=["filter_by_movie", "filter_positive_reviews"],
        )

        output = PipelineOutput(
            type="file",
            path=str(output_path),
            intermediate_dir=str(intermediate_dir),
        )

        pipeline = Pipeline(
            name="q3_count_positive_reviews",
            datasets={"reviews": dataset},
            operations=[movie_filter, positive_filter],
            steps=[step],
            output=output,
            default_model=self.model_name,
        )

        cost = pipeline.run(max_threads=self.concurrent_llm_worker)
        stats = {"cost": cost, "token_usage": {}}

        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        results_df = (
            pd.DataFrame(payload)
            if isinstance(payload, list)
            else pd.DataFrame()
        )

        positive_review_cnt = len(results_df)
        final_df = pd.DataFrame(
            [{"positive_review_cnt": positive_review_cnt}]
        )

        return {"results": final_df, "stats": stats}

    def _execute_q4(self) -> dict:
        """
        Execute Q4: Positivity ratio for movie "taken_3".

        Returns:
            Dict with keys: "results" and "stats"
        """
        try:
            from docetl.api import (
                CodeFilterOp,
                Dataset,
                MapOp,
                Pipeline,
                PipelineOutput,
                PipelineStep,
            )
        except ImportError as e:
            raise RuntimeError(
                "DocETL is not installed in this environment. "
                "Install with: uv pip install docetl 'pyrate-limiter<4'"
            ) from e

        reviews_path = (self.data_path / "Reviews.csv").resolve()
        output_path = (self.results_path / "_docetl_q4_output.json").resolve()
        intermediate_dir = (
            self.results_path / "_docetl_q4_intermediate"
        ).resolve()

        dataset = Dataset(
            type="file",
            path=str(reviews_path),
            source="local",
        )

        movie_filter = CodeFilterOp(
            name="filter_by_movie",
            type="code_filter",
            code=(
                "def transform(doc):\n"
                "    return doc.get('id') == 'taken_3'"
            ),
        )

        add_positivity = MapOp(
            name="add_positivity",
            type="map",
            prompt=(
                "Return 1 if the following review is positive, and 0 if the review "
                "is not positive. Only output a single numeric value (1 or 0) with "
                "no additional commentary\n\nReview: {{ input.reviewText }}"
            ),
            output={"schema": {"positivity": "int"}},
        )

        step = PipelineStep(
            name="calculate_positivity_ratio",
            input="reviews",
            operations=["filter_by_movie", "add_positivity"],
        )

        output = PipelineOutput(
            type="file",
            path=str(output_path),
            intermediate_dir=str(intermediate_dir),
        )

        pipeline = Pipeline(
            name="q4_positivity_ratio",
            datasets={"reviews": dataset},
            operations=[movie_filter, add_positivity],
            steps=[step],
            output=output,
            default_model=self.model_name,
        )

        cost = pipeline.run(max_threads=self.concurrent_llm_worker)
        stats = {"cost": cost, "token_usage": {}}

        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        results_df = (
            pd.DataFrame(payload)
            if isinstance(payload, list)
            else pd.DataFrame()
        )

        if "positivity" in results_df.columns and len(results_df) > 0:
            positivity = pd.to_numeric(
                results_df["positivity"], errors="coerce"
            ).fillna(0.0)
            positivity_ratio = float(positivity.mean())
        else:
            positivity_ratio = 0.0

        final_df = pd.DataFrame([{"positivity_ratio": positivity_ratio}])

        return {"results": final_df, "stats": stats}

    def _execute_q5(self) -> dict:
        """
        Execute Q5: Find review pairs with same sentiment for movie
        "ant_man_and_the_wasp_quantumania".

        Returns:
            Dict with keys: "results" and "stats"
        """
        try:
            from docetl.api import (
                Dataset,
                EquijoinOp,
                Pipeline,
                PipelineOutput,
                PipelineStep,
            )
        except ImportError as e:
            raise RuntimeError(
                "DocETL is not installed in this environment. "
                "Install with: uv pip install docetl 'pyrate-limiter<4'"
            ) from e

        reviews_path = (self.data_path / "Reviews.csv").resolve()
        intermediate_dir = (
            self.results_path / "_docetl_q5_intermediate"
        ).resolve()
        target_pairs = 10

        # Prepare self-join inputs as two named datasets.
        all_reviews = pd.read_csv(reviews_path)
        filtered_reviews = all_reviews[
            all_reviews["id"] == "ant_man_and_the_wasp_quantumania"
        ].copy()

        if len(filtered_reviews) == 0:
            empty = pd.DataFrame(columns=["id", "reviewId", "reviewId2"])
            return {
                "results": empty,
                "stats": {"cost": 0.0, "token_usage": {}},
            }

        intermediate_dir.mkdir(parents=True, exist_ok=True)
        left_path = (intermediate_dir / "q5_left.csv").resolve()
        right_path = (intermediate_dir / "q5_right.csv").resolve()
        filtered_reviews.to_csv(left_path, index=False)
        filtered_reviews.to_csv(right_path, index=False)

        left_dataset = Dataset(
            type="file",
            path=str(left_path),
            source="local",
        )

        right_dataset = Dataset(
            type="file",
            path=str(right_path),
            source="local",
        )

        def normalize_pairs(results_df: pd.DataFrame) -> pd.DataFrame:
            def pick_col(candidates):
                for col in candidates:
                    if col in results_df.columns:
                        return col
                return None

            id_col = pick_col(["id_left", "id:left", "id"])
            left_col = pick_col(
                [
                    "reviewId_left",
                    "reviewId:left",
                    "reviewId1",
                    "reviewId",
                ]
            )
            right_col = pick_col(
                ["reviewId_right", "reviewId:right", "reviewId2"]
            )

            if right_col is None:
                review_cols = [
                    c
                    for c in results_df.columns
                    if "reviewid" in c.lower() and c != left_col
                ]
                if review_cols:
                    right_col = review_cols[0]

            if not (id_col and left_col and right_col and len(results_df) > 0):
                return pd.DataFrame(columns=["id", "reviewId", "reviewId2"])

            projected_df = results_df[[id_col, left_col, right_col]].copy()
            projected_df.columns = ["id", "reviewId", "reviewId2"]
            projected_df = projected_df.dropna(
                subset=["id", "reviewId", "reviewId2"]
            )
            projected_df = projected_df[
                projected_df["reviewId"] != projected_df["reviewId2"]
            ]

            # Remove symmetric duplicates like (a,b) vs (b,a) for the same movie.
            projected_df["_pair_key"] = projected_df.apply(
                lambda row: tuple(
                    sorted([str(row["reviewId"]), str(row["reviewId2"])])
                ),
                axis=1,
            )
            projected_df = projected_df.drop_duplicates(
                subset=["id", "_pair_key"]
            ).drop(columns=["_pair_key"])
            return projected_df.reset_index(drop=True)

        output_path = (self.results_path / "_docetl_q5_output.json").resolve()
        attempt_intermediate_dir = (intermediate_dir / "single_run").resolve()

        join_op = EquijoinOp(
            name="join_same_sentiment",
            type="equijoin",
            comparison_prompt=(
                "These two movie reviews express the same sentiment - either "
                "both are positive or both are negative.\n\n"
                "Review 1: {{ left.reviewText }}\n"
                "Review 2: {{ right.reviewText }}"
            ),
            output={"schema": {"same_sentiment": "bool"}},
        )

        step = PipelineStep(
            name="join_reviews",
            operations=[
                {
                    "join_same_sentiment": {
                        "left": "reviews_left",
                        "right": "reviews_right",
                    }
                }
            ],
        )

        output = PipelineOutput(
            type="file",
            path=str(output_path),
            intermediate_dir=str(attempt_intermediate_dir),
        )

        pipeline = Pipeline(
            name="q5_same_sentiment_pairs",
            datasets={
                "reviews_left": left_dataset,
                "reviews_right": right_dataset,
            },
            operations=[join_op],
            steps=[step],
            output=output,
            default_model=self.model_name,
        )

        cost = pipeline.run(max_threads=self.concurrent_llm_worker)

        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        results_df = (
            pd.DataFrame(payload)
            if isinstance(payload, list)
            else pd.DataFrame()
        )
        projected_df = normalize_pairs(results_df).head(target_pairs).reset_index(
            drop=True
        )
        stats = {"cost": float(cost or 0.0), "token_usage": {}}

        return {"results": projected_df, "stats": stats}

    def _execute_q6(self) -> dict:
        """
        Execute Q6: Find review pairs with opposite sentiment for movie
        "ant_man_and_the_wasp_quantumania".

        Returns:
            Dict with keys: "results" and "stats"
        """
        try:
            from docetl.api import (
                Dataset,
                EquijoinOp,
                Pipeline,
                PipelineOutput,
                PipelineStep,
            )
        except ImportError as e:
            raise RuntimeError(
                "DocETL is not installed in this environment. "
                "Install with: uv pip install docetl 'pyrate-limiter<4'"
            ) from e

        reviews_path = (self.data_path / "Reviews.csv").resolve()
        intermediate_dir = (
            self.results_path / "_docetl_q6_intermediate"
        ).resolve()
        target_pairs = 10

        # Prepare self-join inputs as two named datasets.
        all_reviews = pd.read_csv(reviews_path)
        filtered_reviews = all_reviews[
            all_reviews["id"] == "ant_man_and_the_wasp_quantumania"
        ].copy()

        if len(filtered_reviews) == 0:
            empty = pd.DataFrame(columns=["id", "reviewId", "reviewId2"])
            return {
                "results": empty,
                "stats": {"cost": 0.0, "token_usage": {}},
            }

        intermediate_dir.mkdir(parents=True, exist_ok=True)
        left_path = (intermediate_dir / "q6_left.csv").resolve()
        right_path = (intermediate_dir / "q6_right.csv").resolve()
        filtered_reviews.to_csv(left_path, index=False)
        filtered_reviews.to_csv(right_path, index=False)

        left_dataset = Dataset(
            type="file",
            path=str(left_path),
            source="local",
        )

        right_dataset = Dataset(
            type="file",
            path=str(right_path),
            source="local",
        )

        def normalize_pairs(results_df: pd.DataFrame) -> pd.DataFrame:
            def pick_col(candidates):
                for col in candidates:
                    if col in results_df.columns:
                        return col
                return None

            id_col = pick_col(["id_left", "id:left", "id"])
            left_col = pick_col(
                [
                    "reviewId_left",
                    "reviewId:left",
                    "reviewId1",
                    "reviewId",
                ]
            )
            right_col = pick_col(
                ["reviewId_right", "reviewId:right", "reviewId2"]
            )

            if right_col is None:
                review_cols = [
                    c
                    for c in results_df.columns
                    if "reviewid" in c.lower() and c != left_col
                ]
                if review_cols:
                    right_col = review_cols[0]

            if not (id_col and left_col and right_col and len(results_df) > 0):
                return pd.DataFrame(columns=["id", "reviewId", "reviewId2"])

            projected_df = results_df[[id_col, left_col, right_col]].copy()
            projected_df.columns = ["id", "reviewId", "reviewId2"]
            projected_df = projected_df.dropna(
                subset=["id", "reviewId", "reviewId2"]
            )
            projected_df = projected_df[
                projected_df["reviewId"] != projected_df["reviewId2"]
            ]

            # Remove symmetric duplicates like (a,b) vs (b,a) for the same movie.
            projected_df["_pair_key"] = projected_df.apply(
                lambda row: tuple(
                    sorted([str(row["reviewId"]), str(row["reviewId2"])])
                ),
                axis=1,
            )
            projected_df = projected_df.drop_duplicates(
                subset=["id", "_pair_key"]
            ).drop(columns=["_pair_key"])
            return projected_df.reset_index(drop=True)

        output_path = (self.results_path / "_docetl_q6_output.json").resolve()
        attempt_intermediate_dir = (intermediate_dir / "single_run").resolve()

        join_op = EquijoinOp(
            name="join_opposite_sentiment",
            type="equijoin",
            comparison_prompt=(
                "These two movie reviews express opposite sentiments - one "
                "is positive and the other is negative.\n\n"
                "Review 1: {{ left.reviewText }}\n"
                "Review 2: {{ right.reviewText }}"
            ),
            output={"schema": {"opposite_sentiment": "bool"}},
        )

        step = PipelineStep(
            name="join_reviews",
            operations=[
                {
                    "join_opposite_sentiment": {
                        "left": "reviews_left",
                        "right": "reviews_right",
                    }
                }
            ],
        )

        output = PipelineOutput(
            type="file",
            path=str(output_path),
            intermediate_dir=str(attempt_intermediate_dir),
        )

        pipeline = Pipeline(
            name="q6_opposite_sentiment_pairs",
            datasets={
                "reviews_left": left_dataset,
                "reviews_right": right_dataset,
            },
            operations=[join_op],
            steps=[step],
            output=output,
            default_model=self.model_name,
        )

        cost = pipeline.run(max_threads=self.concurrent_llm_worker)

        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        results_df = (
            pd.DataFrame(payload)
            if isinstance(payload, list)
            else pd.DataFrame()
        )
        projected_df = normalize_pairs(results_df).head(target_pairs).reset_index(
            drop=True
        )
        stats = {"cost": float(cost or 0.0), "token_usage": {}}

        return {"results": projected_df, "stats": stats}

    def _execute_q7(self) -> dict:
        """
        Execute Q7: Find all review pairs with opposite sentiment for movie
        "ant_man_and_the_wasp_quantumania".

        Returns:
            Dict with keys: "results" and "stats"
        """
        try:
            from docetl.api import (
                Dataset,
                EquijoinOp,
                Pipeline,
                PipelineOutput,
                PipelineStep,
            )
        except ImportError as e:
            raise RuntimeError(
                "DocETL is not installed in this environment. "
                "Install with: uv pip install docetl 'pyrate-limiter<4'"
            ) from e

        reviews_path = (self.data_path / "Reviews.csv").resolve()
        intermediate_dir = (
            self.results_path / "_docetl_q7_intermediate"
        ).resolve()

        # Prepare self-join inputs as two named datasets.
        all_reviews = pd.read_csv(reviews_path)
        filtered_reviews = all_reviews[
            all_reviews["id"] == "ant_man_and_the_wasp_quantumania"
        ].copy()

        if len(filtered_reviews) == 0:
            empty = pd.DataFrame(columns=["id", "reviewId", "reviewId2"])
            return {
                "results": empty,
                "stats": {"cost": 0.0, "token_usage": {}},
            }

        intermediate_dir.mkdir(parents=True, exist_ok=True)
        left_path = (intermediate_dir / "q7_left.csv").resolve()
        right_path = (intermediate_dir / "q7_right.csv").resolve()
        filtered_reviews.to_csv(left_path, index=False)
        filtered_reviews.to_csv(right_path, index=False)

        left_dataset = Dataset(
            type="file",
            path=str(left_path),
            source="local",
        )

        right_dataset = Dataset(
            type="file",
            path=str(right_path),
            source="local",
        )

        def normalize_pairs(results_df: pd.DataFrame) -> pd.DataFrame:
            def pick_col(candidates):
                for col in candidates:
                    if col in results_df.columns:
                        return col
                return None

            id_col = pick_col(["id_left", "id:left", "id"])
            left_col = pick_col(
                [
                    "reviewId_left",
                    "reviewId:left",
                    "reviewId1",
                    "reviewId",
                ]
            )
            right_col = pick_col(
                ["reviewId_right", "reviewId:right", "reviewId2"]
            )

            if right_col is None:
                review_cols = [
                    c
                    for c in results_df.columns
                    if "reviewid" in c.lower() and c != left_col
                ]
                if review_cols:
                    right_col = review_cols[0]

            if not (id_col and left_col and right_col and len(results_df) > 0):
                return pd.DataFrame(columns=["id", "reviewId", "reviewId2"])

            projected_df = results_df[[id_col, left_col, right_col]].copy()
            projected_df.columns = ["id", "reviewId", "reviewId2"]
            projected_df = projected_df.dropna(
                subset=["id", "reviewId", "reviewId2"]
            )
            projected_df = projected_df[
                projected_df["reviewId"] != projected_df["reviewId2"]
            ]

            # Remove symmetric duplicates like (a,b) vs (b,a) for the same movie.
            projected_df["_pair_key"] = projected_df.apply(
                lambda row: tuple(
                    sorted([str(row["reviewId"]), str(row["reviewId2"])])
                ),
                axis=1,
            )
            projected_df = projected_df.drop_duplicates(
                subset=["id", "_pair_key"]
            ).drop(columns=["_pair_key"])
            return projected_df.reset_index(drop=True)

        output_path = (self.results_path / "_docetl_q7_output.json").resolve()
        attempt_intermediate_dir = (intermediate_dir / "single_run").resolve()

        join_op = EquijoinOp(
            name="join_opposite_sentiment",
            type="equijoin",
            comparison_prompt=(
                "These two movie reviews express opposite sentiments - one "
                "is positive and the other is negative.\n\n"
                "Review 1: {{ left.reviewText }}\n"
                "Review 2: {{ right.reviewText }}"
            ),
            output={"schema": {"opposite_sentiment": "bool"}},
        )

        step = PipelineStep(
            name="join_reviews",
            operations=[
                {
                    "join_opposite_sentiment": {
                        "left": "reviews_left",
                        "right": "reviews_right",
                    }
                }
            ],
        )

        output = PipelineOutput(
            type="file",
            path=str(output_path),
            intermediate_dir=str(attempt_intermediate_dir),
        )

        pipeline = Pipeline(
            name="q7_all_opposite_sentiment_pairs",
            datasets={
                "reviews_left": left_dataset,
                "reviews_right": right_dataset,
            },
            operations=[join_op],
            steps=[step],
            output=output,
            default_model=self.model_name,
        )

        cost = pipeline.run(max_threads=self.concurrent_llm_worker)

        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        results_df = (
            pd.DataFrame(payload)
            if isinstance(payload, list)
            else pd.DataFrame()
        )
        projected_df = normalize_pairs(results_df)
        stats = {"cost": float(cost or 0.0), "token_usage": {}}

        return {"results": projected_df, "stats": stats}

    def _execute_q8(self) -> dict:
        """
        Execute Q8: Calculate the number of positive and negative reviews
        for movie "taken_3".

        Returns:
            Dict with keys: "results" and "stats"
        """
        try:
            from docetl.api import (
                CodeFilterOp,
                Dataset,
                MapOp,
                Pipeline,
                PipelineOutput,
                PipelineStep,
            )
        except ImportError as e:
            raise RuntimeError(
                "DocETL is not installed in this environment. "
                "Install with: uv pip install docetl 'pyrate-limiter<4'"
            ) from e

        reviews_path = (self.data_path / "Reviews.csv").resolve()
        output_path = (self.results_path / "_docetl_q8_output.json").resolve()
        intermediate_dir = (
            self.results_path / "_docetl_q8_intermediate"
        ).resolve()

        dataset = Dataset(
            type="file",
            path=str(reviews_path),
            source="local",
        )

        movie_filter = CodeFilterOp(
            name="filter_by_movie",
            type="code_filter",
            code=(
                "def transform(doc):\n"
                "    return doc.get('id') == 'taken_3'"
            ),
        )

        add_sentiment = MapOp(
            name="add_sentiment",
            type="map",
            prompt=(
                "Return POSITIVE if the following review is positive, and "
                "NEGATIVE if the review is not positive. Only output "
                "POSITIVE or NEGATIVE with no additional commentary\n\n"
                "Review: {{ input.reviewText }}"
            ),
            output={"schema": {"sentiment": "str"}},
        )

        step = PipelineStep(
            name="count_sentiments",
            input="reviews",
            operations=["filter_by_movie", "add_sentiment"],
        )

        output = PipelineOutput(
            type="file",
            path=str(output_path),
            intermediate_dir=str(intermediate_dir),
        )

        pipeline = Pipeline(
            name="q8_sentiment_counts",
            datasets={"reviews": dataset},
            operations=[movie_filter, add_sentiment],
            steps=[step],
            output=output,
            default_model=self.model_name,
        )

        cost = pipeline.run(max_threads=self.concurrent_llm_worker)
        stats = {"cost": float(cost or 0.0), "token_usage": {}}

        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        results_df = (
            pd.DataFrame(payload)
            if isinstance(payload, list)
            else pd.DataFrame()
        )

        if "sentiment" not in results_df.columns or len(results_df) == 0:
            final_df = pd.DataFrame(
                [
                    {"sentiment": "NEGATIVE", "count": 0},
                    {"sentiment": "POSITIVE", "count": 0},
                ]
            )
            return {"results": final_df, "stats": stats}

        normalized_sentiment = (
            results_df["sentiment"].astype(str).str.strip().str.upper()
        )

        def normalize_label(label: str) -> str:
            if "POSITIVE" in label:
                return "POSITIVE"
            if "NEGATIVE" in label:
                return "NEGATIVE"
            return "NEGATIVE"

        normalized_sentiment = normalized_sentiment.apply(normalize_label)

        sentiment_counts = (
            normalized_sentiment.value_counts().rename_axis("sentiment")
            .reset_index(name="count")
        )

        for sentiment in ["POSITIVE", "NEGATIVE"]:
            if sentiment not in sentiment_counts["sentiment"].values:
                sentiment_counts = pd.concat(
                    [
                        sentiment_counts,
                        pd.DataFrame(
                            [{"sentiment": sentiment, "count": 0}]
                        ),
                    ],
                    ignore_index=True,
                )

        final_df = sentiment_counts.sort_values("sentiment").reset_index(
            drop=True
        )

        return {"results": final_df[["sentiment", "count"]], "stats": stats}

    def _execute_q9(self) -> dict:
        """
        Execute Q9: Score from 1 to 5 how much did the reviewer like the
        movie "ant_man_and_the_wasp_quantumania".

        Returns:
            Dict with keys: "results" and "stats"
        """
        try:
            from docetl.api import (
                CodeFilterOp,
                Dataset,
                MapOp,
                Pipeline,
                PipelineOutput,
                PipelineStep,
            )
        except ImportError as e:
            raise RuntimeError(
                "DocETL is not installed in this environment. "
                "Install with: uv pip install docetl 'pyrate-limiter<4'"
            ) from e

        reviews_path = (self.data_path / "Reviews.csv").resolve()
        output_path = (self.results_path / "_docetl_q9_output.json").resolve()
        intermediate_dir = (
            self.results_path / "_docetl_q9_intermediate"
        ).resolve()

        dataset = Dataset(
            type="file",
            path=str(reviews_path),
            source="local",
        )

        movie_filter = CodeFilterOp(
            name="filter_by_movie",
            type="code_filter",
            code=(
                "def transform(doc):\n"
                "    return doc.get('id') == 'ant_man_and_the_wasp_quantumania'"
            ),
        )

        add_review_score = MapOp(
            name="add_review_score",
            type="map",
            prompt=(
                "Score from 1 to 5 how much did the reviewer like the movie "
                "based on provided rubrics.\n\n"
                "Rubrics:\n"
                "5: Very positive. Strong positive sentiment, indicating high "
                "satisfaction.\n"
                "4: Positive. Noticeably positive sentiment, indicating general "
                "satisfaction.\n"
                "3: Neutral. Expresses no clear positive or negative sentiment. "
                "May be factual or descriptive without emotional language.\n"
                "2: Negative. Noticeably negative sentiment, indicating some "
                "level of dissatisfaction but without strong anger or "
                "frustration.\n"
                "1: Very negative. Strong negative sentiment, indicating high "
                "dissatisfaction, frustration, or anger.\n\n"
                "Review: {{ input.reviewText }}\n\n"
                "Only provide the score number (1-5) with no other comments."
            ),
            output={"schema": {"reviewScore": "int"}},
        )

        step = PipelineStep(
            name="score_reviews",
            input="reviews",
            operations=["filter_by_movie", "add_review_score"],
        )

        output = PipelineOutput(
            type="file",
            path=str(output_path),
            intermediate_dir=str(intermediate_dir),
        )

        pipeline = Pipeline(
            name="q9_review_scores",
            datasets={"reviews": dataset},
            operations=[movie_filter, add_review_score],
            steps=[step],
            output=output,
            default_model=self.model_name,
        )

        cost = pipeline.run(max_threads=self.concurrent_llm_worker)
        stats = {"cost": float(cost or 0.0), "token_usage": {}}

        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        results_df = (
            pd.DataFrame(payload)
            if isinstance(payload, list)
            else pd.DataFrame()
        )

        if "reviewId" not in results_df.columns:
            empty = pd.DataFrame(columns=["reviewId", "reviewScore"])
            return {"results": empty, "stats": stats}

        # Keep ranking outputs numeric and bounded for stable evaluation.
        review_scores = pd.to_numeric(
            results_df.get("reviewScore"), errors="coerce"
        ).fillna(3.0)
        review_scores = review_scores.clip(lower=1.0, upper=5.0)

        final_df = pd.DataFrame(
            {
                "reviewId": results_df["reviewId"],
                "reviewScore": review_scores,
            }
        )

        return {"results": final_df[["reviewId", "reviewScore"]], "stats": stats}

    def _execute_q10(self) -> dict:
        """
        Execute Q10: Rank movies based on review scores.

        Returns:
            Dict with keys: "results" and "stats"
        """
        try:
            from docetl.api import (
                Dataset,
                MapOp,
                Pipeline,
                PipelineOutput,
                PipelineStep,
            )
        except ImportError as e:
            raise RuntimeError(
                "DocETL is not installed in this environment. "
                "Install with: uv pip install docetl 'pyrate-limiter<4'"
            ) from e

        reviews_path = (self.data_path / "Reviews.csv").resolve()
        output_path = (self.results_path / "_docetl_q10_output.json").resolve()
        intermediate_dir = (
            self.results_path / "_docetl_q10_intermediate"
        ).resolve()

        dataset = Dataset(
            type="file",
            path=str(reviews_path),
            source="local",
        )

        add_review_score = MapOp(
            name="add_review_score",
            type="map",
            prompt=(
                "Score from 1 to 5 how much did the reviewer like the movie "
                "based on provided rubrics.\n\n"
                "Rubrics:\n"
                "5: Very positive. Strong positive sentiment, indicating high "
                "satisfaction.\n"
                "4: Positive. Noticeably positive sentiment, indicating general "
                "satisfaction.\n"
                "3: Neutral. Expresses no clear positive or negative sentiment. "
                "May be factual or descriptive without emotional language.\n"
                "2: Negative. Noticeably negative sentiment, indicating some "
                "level of dissatisfaction but without strong anger or "
                "frustration.\n"
                "1: Very negative. Strong negative sentiment, indicating high "
                "dissatisfaction, frustration, or anger.\n\n"
                "Review: {{ input.reviewText }}\n\n"
                "Only provide the score number (1-5) with no other comments."
            ),
            output={"schema": {"reviewScore": "int"}},
        )

        step = PipelineStep(
            name="rank_movies",
            input="reviews",
            operations=["add_review_score"],
        )

        output = PipelineOutput(
            type="file",
            path=str(output_path),
            intermediate_dir=str(intermediate_dir),
        )

        pipeline = Pipeline(
            name="q10_movie_rankings",
            datasets={"reviews": dataset},
            operations=[add_review_score],
            steps=[step],
            output=output,
            default_model=self.model_name,
        )

        cost = pipeline.run(max_threads=self.concurrent_llm_worker)
        stats = {"cost": float(cost or 0.0), "token_usage": {}}

        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        results_df = (
            pd.DataFrame(payload)
            if isinstance(payload, list)
            else pd.DataFrame()
        )

        if "id" not in results_df.columns:
            empty = pd.DataFrame(columns=["movieId", "movieScore"])
            return {"results": empty, "stats": stats}

        review_scores = pd.to_numeric(
            results_df.get("reviewScore"), errors="coerce"
        ).fillna(3.0)
        review_scores = review_scores.clip(lower=1.0, upper=5.0)

        scored_reviews = pd.DataFrame(
            {
                "movieId": results_df["id"],
                "reviewScore": review_scores,
            }
        ).dropna(subset=["movieId"])

        if len(scored_reviews) == 0:
            empty = pd.DataFrame(columns=["movieId", "movieScore"])
            return {"results": empty, "stats": stats}

        movie_scores = scored_reviews.groupby("movieId", as_index=False)[
            "reviewScore"
        ].mean()
        movie_scores = movie_scores.rename(
            columns={"reviewScore": "movieScore"}
        )

        return {
            "results": movie_scores[["movieId", "movieScore"]],
            "stats": stats,
        }