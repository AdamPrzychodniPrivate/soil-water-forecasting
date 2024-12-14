from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model, extract_target_variable, extract_covariates, generate_metadata_array


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="regressor",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                name="evaluate_model_node",
                outputs="metrics",
            ),
            node(
                func=extract_target_variable,
                inputs=["ds_features", "params:target"],
                outputs=["target_data", "target_mask"],
                name="extract_target_node"
            ),
            node(
                func=extract_covariates,
                inputs=["ds_features", "params:covariates"],
                outputs="covariates_data",
                name="extract_covariates_node"
            ),
            node(
                func=generate_metadata_array,
                inputs="ds_features",
                outputs="metadata_array",
                name="extract_metadata_node"
            ),
        ]
    )
