from typing import List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from GPy.kern import RBF, Bias
from GPy.models import GPLVM, BayesianGPLVM, SparseGPLVM


## PCA
def pca_graphable(df, column_names):
    components = PCA(n_components=2).fit_transform(df[column_names].to_numpy())
    df["Component1"] = components[:, 0]
    df["Component2"] = components[:, 1]
    return df


def principle_component_analysis(df, column_names, n_components):
    pca = PCA(n_components=n_components)
    pca.fit_transform(df[column_names].to_numpy())
    return pca


## GPy LVMs
def gp_model(
    df,
    n_components=2,
    sort_by: List[str] = ["Region", "Subject", "Feature"],
    expand_dims: bool = False,
    optimize: bool = True,
):
    data = df.sort_values(
        by=sort_by,
    ).to_numpy()
    if expand_dims:
        M = np.random.normal(0, 1e-6, size=(data.shape[1], 100))
        data = data * M
    model = GPLVM(
        data,
        n_components,
        kernel=RBF(n_components, variance=1, lengthscale=0.2, ARD=True)
        + Bias(n_components),
    )
    model.likelihood.variance = 1.0
    if optimize:
        model.optimize(messages=1, max_iters=5e4)
    return model


def sparse_gp_model(
    df,
    n_components=2,
    n_inducing=15,
    sort_by: List[str] = ["Region", "Subject", "Feature"],
    expand_dims: bool = False,
    optimize: bool = True,
):
    data = df.sort_values(
        by=sort_by,
    ).to_numpy()
    if expand_dims:
        M = np.random.normal(0, 1e-6, size=(data.shape[1], 100))
        data = data * M
    model = SparseGPLVM(
        data,
        n_components,
        num_inducing=n_inducing,
        kernel=RBF(n_components, variance=1, lengthscale=0.2, ARD=True)
        + Bias(n_components),
    )
    model.likelihood.variance = 1.0
    if optimize:
        model.optimize(messages=1, max_iters=5e4)
    return model


def bayesian_gp_model(
    df,
    n_components=10,
    n_inducing=15,
    sort_by: List[str] = ["Region", "Subject", "Feature"],
    expand_dims: bool = False,
    optimize: bool = True,
):
    data = df.sort_values(
        by=sort_by,
    ).to_numpy()
    if expand_dims:
        M = np.random.normal(0, 1e-6, size=(data.shape[1], 100))
        data = data * M
    model = BayesianGPLVM(
        data,
        n_components,
        num_inducing=n_inducing,
        kernel=RBF(n_components, variance=1, lengthscale=0.2, ARD=True)
        + Bias(n_components),
    )
    model.likelihood.variance = 1.0
    if optimize:
        model.optimize(messages=1, max_iters=5e4)
    return model


def get_gp_covariance(
    df, model, sort_by: List[str] = ["Region", "Feature", "Subject"]
):
    kernel = model.kern.K(model.X)
    data = df.reset_index().sort_values(
        by=sort_by,
        ignore_index=True,
    )
    labels = list(zip(*[data[reg] for reg in sort_by]))
    index = pd.MultiIndex.from_tuples(labels, names=sort_by)
    correlation = pd.DataFrame(kernel, index=index, columns=index)
    correlation = correlation.reorder_levels(sort_by, axis=0)
    correlation = correlation.reorder_levels(sort_by, axis=1)
    return correlation, labels


def create_named_gp(
    df, model_name: str, n_components: int, sort_by, optimize: bool = True
):
    match model_name:
        case "model":
            return gp_model(
                df,
                n_components=n_components,
                sort_by=sort_by,
                expand_dims=False,
                optimize=optimize,
            )
        case "modified_model":
            return gp_model(
                df,
                n_components=n_components,
                sort_by=sort_by,
                expand_dims=True,
                optimize=optimize,
            )
        case "bayesian_model":
            return bayesian_gp_model(
                df,
                n_components=n_components,
                sort_by=sort_by,
                expand_dims=False,
                optimize=optimize,
            )
        case "modified_bayesian_model":
            return bayesian_gp_model(
                df,
                n_components=n_components,
                sort_by=sort_by,
                expand_dims=True,
                optimize=optimize,
            )
        case "sparse_model":
            return sparse_gp_model(
                df,
                n_components=n_components,
                sort_by=sort_by,
                expand_dims=False,
                optimize=optimize,
            )
        case "modified_sparse_model":
            return sparse_gp_model(
                df,
                n_components=n_components,
                sort_by=sort_by,
                expand_dims=True,
                optimize=optimize,
            )
        case _:
            return gp_model(
                df,
                n_components=n_components,
                sort_by=sort_by,
                expand_dims=True,
                optimize=optimize,
            )


def load_gp_model(model, file):
    model.update_model(
        False
    )  # do not call the underlying expensive algebra on load
    model.initialize_parameter()  # Initialize the parameters (connect the parameters up)
    model[:] = np.load(file)  # Load the parameters
    model.update_model(True)  # Call the algebra only once
    return model


def save_gp_model(model, file):
    np.save(file, model.param_array)
