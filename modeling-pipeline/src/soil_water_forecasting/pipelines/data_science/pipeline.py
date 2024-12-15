from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model, extract_target_variable, extract_covariates, generate_metadata_array, create_distance_matrix


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
            node(
                func=create_distance_matrix,
                inputs="metadata_array",
                outputs="distance_matrix",
                name="extract_distance_matrix_node"
            ),
            node(
                func=get_connectivity_matrix,
                inputs=["target_data", 
                        "target_mask", 
                        "distance_matrix",
                        "covariates_data",
                        "metadata_array",
                        "params:connectivity.method",
                        "params:connectivity.threshold",
                        "params:connectivity.knn",
                        "params:connectivity.binary_weights",
                        "params:connectivity.include_self",
                        "params:connectivity.force_symmetric",
                        "params:connectivity.layout",
                        ],
                outputs="connectivity_matrix",
                name="extract_connectivity_node"
            ),
        ]
    )
