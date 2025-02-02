from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_model_input_table, preprocess_companies, preprocess_shuttles, preprocess, feature_engineering


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_companies,
                inputs="companies",
                outputs=["preprocessed_companies", "companies_columns"],
                name="preprocess_companies_node",
            ),
            node(
                func=preprocess_shuttles,
                inputs="shuttles",
                outputs="preprocessed_shuttles",
                name="preprocess_shuttles_node",
            ),
            node(
                func=create_model_input_table,
                inputs=["preprocessed_shuttles", "preprocessed_companies", "reviews"],
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
            node(
                func=preprocess,
                inputs=[
                    "ds_0",
                    "ds_1",
                    "ds_2",
                    "params:preprocess.target_lat",
                    "params:preprocess.target_lon",
                    "params:preprocess.method",
                    "params:preprocess.fill_variables",
                    "params:preprocess.fill_value"
                ],
                outputs="ds_preprocessed",
                name="preprocess_data_node",
            ),
            node(
                func=feature_engineering,
                inputs=["ds_preprocessed"],
                outputs="ds_features",
                name="feature_engineering_node",
            ),
        ]
    )