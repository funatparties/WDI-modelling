#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 19:10:57 2021

@author: JoshM
"""

from data_collection import load_data, INDICATORS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

#Functions used for tick label formatting on heatmaps
NO_FMT = lambda x:x
EXP_FMT = lambda x:'{:.1e}'.format(x)

#TODO: split dataset loading and preprocessing into new file

def get_ind_name(key):
    """Gets the full name for an indicator based on its code if the code is
    found in data_collection.INDICATORS.

    Parameters
    ----------
    key : str
        API code for indicator. By default, all indicator fields in the data
        are named with API codes.

    Returns
    -------
    str
        Full name of the indicator, or simply returns input if it is not found.

    """
    
    return INDICATORS.get(key, key)

def X_y_split(df, label_name):
    """Convenience function for splitting a dataframe into a design matrix and
    prediction targets.

    Parameters
    ----------
    df : DataFrame
        Dataframe to be split.
    label_name : str
        The name of the column containing prediction targets.

    Returns
    -------
    DataFrame
        The design matrix X containing the independent variables for modelling.
    Series
        The labels or dependent variable for modelling.

    """
    
    return df.drop(columns=[label_name]), df[label_name]

def full_ds():
    """Returns the full stored data set with human-readable indicator names
    applied.

    Returns
    -------
    df : DataFrame
        The dataframe containg the dataset.

    """
    
    df = load_data()
    df = df.rename(columns = lambda x:get_ind_name(x))
    return df

def numeric_lending_ds():
    """Returns a design matrix of numeric training data and a set of class 
    labels for predicting Lending Type of countries. Some preprocessing is 
    applied such as dropping irrelevant or non-numeric columns, and applying 
    log transformations to certain others.

    Returns
    -------
    DataFrame
        The design matrix X containing numeric training data.
    Series
        The Lending Type class labels for each country.

    """
    
    df = load_data()
    # Drop unused columns
    df = df.drop(columns=['ID','Name','Region','Income Level','FP.CPI.TOTL.ZG',
                          'EG.USE.ELEC.KH.PC'])
    # Drop rows with missing values
    df = df.dropna()
    # These columns will be transformed with logs
    log_columns = ['NY.GDP.MKTP.CD','NY.GDP.PCAP.CD','SP.POP.TOTL']
    df[log_columns] = np.log(df[log_columns])
    # Renaming of only particular columns seems unsupported by pandas.
    # May as well append log indicators while translating codes into names.
    df.columns = ['log '+get_ind_name(name) if (name in log_columns) else 
                  get_ind_name(name) for name in df.columns]
    return X_y_split(df,'Lending Type')

#TODO: add imputation for missing values

def exploratory_plot(df,x_field,y_field,label_field,**kwargs):
    """Convenience function for producing scatter plots of pairs of variables
    with points labelled according to categories.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing the data and labels for plotting.
    x_field : str
        Name of the column to be plotted on the x-axis.
    y_field : str
        Name of the column to be plotted on the y-axis.
    label_field : str
        Name of the column to be used for labelling points.
    **kwargs : dict-like
        Kwargs to be passed to axes, such as xscale='log'.

    Returns
    -------
    None.

    """
    
    x = df[x_field]
    y = df[y_field]
    labels = df[label_field]
    fig,ax = plt.subplots(dpi=200, subplot_kw=kwargs)
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    for label in labels.unique():
        ax.scatter(x[labels == label],y[labels == label],label=label)
    ax.legend()
    plt.show()
    return


def RBF_SVM_gridsearch(X, y, C_range = np.logspace(-3,3,7),
                 gamma_range = np.logspace(-3,3,7),
                 plot = True, **kwargs):
    """Trains a support vector machine with a radial basis kernel on the 
    supplied data using 2D grid search to optimise the regularisation and 
    length scale parameters.
    

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The design matrix of the training data.
    y : array-like of shape (n_samples,)
        The class labels of the data.
    C_range : array-like of shape (n_values,), optional
        The list of values to try for the regularisation parameter C. This
        parameter controls the penalty weight of misclassifications against
        maximising margin size. Larger values prioritise correct 
        classifications. The default is np.logspace(-3,3,7).
    gamma_range : array-like of shape (n_values,), optional
        The list of values to try for the kernel parameter gamma. This 
        parameter functions like the inverse of the length scale of the 
        Gaussian basis function. Smaller values cause wider functions and give
        each point a larger radius of influence, resulting in coarser decision
        boundaries. The default is np.logspace(-3,3,7).
    plot : bool, optional
        If True, will plot a heatmap and slice validation curves over C and
        gamma. If there are only two variables to the data, decision boundaries
        will also be visualised. The default is True.
    **kwargs : dict
        Kwargs to be passed to SVC().

    Returns
    -------
    GridSearchCV
        The grid search object containing the results of the parameter sweep
        as well as the best estimator refit on the full training data.

    """
    
    # Build classifier
    param_grid = {'svc__C':C_range, 'svc__gamma':gamma_range}
    model = Pipeline(steps=[('scaler',StandardScaler()),
                           ('svc',SVC(kernel='rbf',**kwargs))])
    grid = GridSearchCV(model, param_grid=param_grid,
                        cv=5,return_train_score=True)
    # Train classifiers
    grid.fit(X,y)
    # Print best
    print("Maximum mean validation score was {0:.3f} with parameters {1}".format(
        grid.best_score_,grid.best_params_))
    
    if plot:
        if X.shape[1] == 2:
            # Two variables, so plot decision boundaries
            fig, axs = SVM_summary_2D(grid, param_grid, X, y)
            plt.show()
        else:
            # Plot heatmap and slices separately
            # Heatmap
            fig,ax = plt.subplots(dpi=200)
            plot_RBF_SVM_heatmap(ax, grid, param_grid)
            plt.show()
            
            # Slices
            fig,axs = plt.subplots(1,2,dpi=200,figsize=(12,6),sharey=True)
            plot_RBF_SVM_slices(axs, grid, param_grid)
            plt.show()
    return grid

def linear_SVM(X, y, C_range = np.logspace(-3,3,7), plot = True, **kwargs):
    """Trains a linear support vector machine on the supplied data using 1D
    grid search to optimise the regularisation parameter.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The design matrix of the training data.
    y : array-like of shape (n_samples,)
        The class labels of the data.
    C_range : array-like of shape (n_values,), optional
        The list of values to try for the regularisation parameter C. This
        parameter controls the penalty weight of misclassifications against
        maximising margin size. Larger values prioritise correct 
        classifications. The default is np.logspace(-3,3,7).
    plot : bool, optional
        If True, will plot a validation curve over C. The default is True.
    **kwargs : dict
        Kwargs to be passed to LinearSVC().

    Returns
    -------
    GridSearchCV
        The grid search object containing the results of the parameter sweep
        as well as the best estimator refit on the full training data.

    """
    
    # Build classifier
    param_grid = {'svc__C':C_range}
    # LinearSVC() is faster than SVC() with linear kernel but uses slightly
    # different settings.
    model = Pipeline(steps=[('scaler',StandardScaler()),
                            ('svc',LinearSVC(**kwargs))])
    grid = GridSearchCV(model, param_grid=param_grid,
                        cv=5, return_train_score = True)
    # Train classifiers
    grid.fit(X,y)

    # Print max
    print("Maximum mean validation score was {0:.3f} with parameters {1}".format(
        grid.best_score_,grid.best_params_))
    
    if plot:
        # Extract scores
        val_scores = grid.cv_results_["mean_test_score"]
        val_stds = grid.cv_results_["std_test_score"]
        train_scores = grid.cv_results_["mean_train_score"]
        # Plot curve
        fig,ax = plt.subplots(dpi=200)
        ax.set_title("Linear SVC Validation Curve")
        ax.set_xlabel("C")
        ax.set_ylabel("Accuracy")
        ax.set_xscale('log')
        plot_validation_curve(ax, C_range, train_scores, val_scores, val_stds)
        ax.legend()
        plt.show()
    return grid

def SVM_summary_2D(grid, param_grid, X, y, **kwargs):
    """Produces a 2 x 2 array of plots summarising a support vector machine 
    grid search on 2-dimensional data. The array consists of a heatmap,
    validation curves on slices of the grid search, and a countour plot 
    visualising the decision boundaries of the final model. New figure and axes
    are created but not shown.

    Parameters
    ----------
    grid : GridSearchCV
        Object representing the grid search containing results to be plotted.
    param_grid : dict with form {param_name:[param_values,]}
        Dict of the parameter grid the search was over.
    X : array-like of shape (n_samples, n_features)
        The design matrix of the training data.
    y : array-like of shape (n_samples,)
        The class labels of the data.
    **kwargs : dict
        Kwargs to be passed to subplots().

    Returns
    -------
    fig : Figure
        The figure containing the axes.
    axs : array of Axes of shape (2,2)
        The array of axes containing the plots.

    """
    
    # 2 x 2 plot grid
    fig,axs = plt.subplots(2,2, dpi=200, figsize=(12,12), **kwargs)
    # Heatmap upper left
    plot_RBF_SVM_heatmap(axs[0,0], grid, param_grid)
    # Slices lower half
    plot_RBF_SVM_slices(axs[1,:], grid, param_grid)
    # Counters upper right
    plot_clf_contours(axs[0,1], grid.best_estimator_, X, y)
    axs[0,1].set_title("SVC Decision Boundaries")
    axs[0,1].set_xlabel(X.columns[0])
    axs[0,1].set_ylabel(X.columns[1])
    return fig, axs


def visualise_clf(model, X, y, cmap=plt.cm.coolwarm):
    """Convenience function producing a stand-alone contour plot for a
    classifier.

    Parameters
    ----------
    model : Classifier object with predict() method
        The classifier to visualise.
    X : array-like
        The design matrix of the data.
    y : array-like
        The class labels for the data.
    cmap : cmap, optional
        The colourmap used for the contour plot.
        The default is plt.cm.coolwarm.

    Returns
    -------
    None.

    """
    
    fig,ax = plt.subplots(dpi=200)
    plot_clf_contours(ax, model, X, y, cmap=cmap)
    ax.legend()
    plt.show()
    return

def plot_RBF_SVM_heatmap(ax, grid, param_grid):
    """Convenience function for plotting grid heatmap with preset formatting
    appropriate for RBF support vector machine gridsearch. Adds the plot to 
    supplied Axes without showing figure so that Axes can be customised.

    Parameters
    ----------
    axs : Axes
        The axes to plot the heatmap on.
    grid : GridSearchCV
        Object representing the grid search containing results to be plotted.
    param_grid : dict with form {param_name:[param_values,]}
        Dict of the parameter grid the search was over.

    Returns
    -------
    None.

    """
    plot_grid_heatmap(ax, grid, param_grid,
                      x_fmt = EXP_FMT, y_fmt = EXP_FMT)
    ax.set_xlabel("Gamma")
    ax.set_ylabel("C")
    ax.set_title("RBF SVC Grid Search Heatmap")
    return
    
def plot_RBF_SVM_slices(axs, grid, param_grid):
    """Convenience function for plotting grid slices with preset formatting
    appropriate for RBF support vector machine gridsearch. Adds the plot to 
    supplied Axes without showing figure so that Axes can be customised.

    Parameters
    ----------
    axs : [Axes,Axes]
        A pair of axes to plot the valdiation curves on. Works with subplot
        slices.
    grid : GridSearchCV
        Object representing the grid search containing results to be plotted.
    param_grid : dict with form {param_name:[param_values,]}
        Dict of the parameter grid the search was over.

    Returns
    -------
    None.

    """
    
    axs[1].tick_params(labelleft=True)
    plot_grid_slices(axs, grid, param_grid)
    axs[0].set_title("Validation Curve for C = {0:.1e}".format(
        grid.best_params_['svc__C']))
    axs[0].set_xlabel("Gamma")
    axs[0].set_xscale('log')
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    axs[1].set_title("Validation Curve for Gamma = {0:.1e}".format(
        grid.best_params_['svc__gamma']))
    axs[1].set_xlabel("C")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_xscale('log')
    axs[1].legend()
    return

#TODO: Add option for automatic axis labelling using param_grid names

def plot_grid_heatmap(ax, grid, param_grid, cmap = plt.cm.plasma,
                      x_fmt = NO_FMT, y_fmt = NO_FMT):
    """Plots grid search results as a heatmap showing relative validation
    accuracies over parameter values. Only works if the gridsearch is over two
    parameters and the grid must already be fit.
    
    Adds the plot to supplied Axes without showing figure so that Axes can be
    customised. Tick label formatting must be done via functions passed in as 
    arguments due to weirdness with custom ticks and imshow().

    Parameters
    ----------
    ax : Axes
        The axes to plot the heatmap on.
    grid : GridSearchCV
        Object representing the grid search containing results to be plotted.
    param_grid : dict with form {param_name:[param_values,]}
        Dict of the parameter grid the search was over.
    cmap : cmap, optional
        Colour map used for the heatmap. The default is plt.cm.plasma.
    x_fmt : func with signature func(float) -> str, optional
        Function for formatting tick labels on x-axis. The default is NO_FMT.
    y_fmt :func with signature func(float) -> str, optional
        Function for formatting tick labels on y-axis. The default is NO_FMT.

    Returns
    -------
    None.

    """
    # Extract parameter names and ranges
    (var1_name, var1_range), (var2_name, var2_range) = param_grid.items()
    # Shape validation scores into grid
    scores = grid.cv_results_["mean_test_score"].reshape(len(var1_range),
                                                     len(var2_range))
    im = ax.imshow(scores,cmap=cmap)
    # Setting tick positions to correspond to pixel locations i.e. [0,1,2,...]
    ax.set_xticks(np.arange(len(var2_range)))
    ax.set_yticks(np.arange(len(var1_range)))
    # Must manually set tick labels using parameter values. MPL Formatters do
    # not seem to work due to mismatch between label values and positions or
    # some other weirdness with imshow().
    ax.set_xticklabels(map(x_fmt,var2_range),rotation=90) 
    ax.set_yticklabels(map(y_fmt,var1_range))
    plt.colorbar(mappable=im,label="Validation Accuracy",ax=ax)
    return

def plot_grid_slices(axs, grid, param_grid):
    """Plots a pair of validation curves representing results from taking
    slices through the grid search which intersect the best value. That is, for
    each parameter, plot the validation curve obtained by holding it fixed at 
    its optimal value while sweeping the other parameter. Only configured for
    grid with two parameters.

    Parameters
    ----------
    axs : [Axes,Axes]
        A pair of axes to plot the valdiation curves on. Works with subplot
        slices.
    grid : GridSearchCV
        Object representing the grid search containing results to be plotted.
    param_grid : dict with form {param_name:[param_values,]}
        Dict of the parameter grid the search was over.

    Returns
    -------
    None.

    """
    
    # Extract parameter names and ranges
    (var1_name, var1_range), (var2_name, var2_range) = param_grid.items()
    # Extract coordinates of best values
    var1_best = grid.best_params_[var1_name]
    var2_best = grid.best_params_[var2_name]
    var1_idx = np.where(var1_range == var1_best)
    var2_idx = np.where(var2_range == var2_best)
    # Shape score values into grid shape matching heatmap
    val_scores = grid.cv_results_["mean_test_score"].reshape(len(var1_range),
                                                     len(var2_range))
    val_stds = grid.cv_results_["std_test_score"].reshape(len(var1_range),
                                                     len(var2_range))
    train_scores = grid.cv_results_["mean_train_score"].reshape(len(var1_range),
                                                     len(var2_range))
    # Take a horizontal slice through the grid which intersects the best value
    # and plot a validation curve of those scores
    axs[0].set_xticks(var2_range)
    plot_validation_curve(axs[0], var2_range, train_scores[var1_idx,:].flatten(),
                          val_scores[var1_idx,:].flatten(),
                          val_stds[var1_idx,:].flatten())
    # Take a vertical slice through the grid which intersects the best value
    # and plot a validation curve of those scores
    axs[1].set_xticks(var1_range)
    plot_validation_curve(axs[1], var1_range, train_scores[:,var2_idx].flatten(),
                          val_scores[:,var2_idx].flatten(),
                          val_stds[:,var2_idx].flatten())
    return

def plot_validation_curve(ax, X, train_mean, val_mean, val_std):
    """Plots a validation curve showing training and validation scores over
    a range of parameter values. Adds the plot to supplied Axes without showing
    figure so that Axes can be customised.

    Parameters
    ----------
    ax : Axes
        The axes to plot the curve on.
    X : array-like of shape (n_values,)
        The list of parameter values corresponding to the scores.
    train_mean : array-like
        The mean training scores for each parameter value.
    val_mean : array-like
        The mean validation scores for each parameter value.
    val_std : array-like
        The standard deviation of validation scores for each parameter value.

    Returns
    -------
    None.

    """
    
    # Training score
    ax.plot(X, train_mean,'b-',label="Training score")
    # Validation score with error bars of 1 standard deviation 
    ax.errorbar(X, val_mean, val_std, fmt='r-',capsize=3,
                label="Validation score")
    return

def grid_bounds(x, bleed=0.2):
    """Produces bounds for generating a meshgrid used in contour plots. The
    bounds add a fixed proportion (given by bleed) of the total data range on
    either side of the data extremes.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        Data of the dimension for which grid bounds are to be generated.
    bleed : float, optional
        The proportion of the data range to add on either side of the extremes.
        The default is 0.2.

    Returns
    -------
    float
        Minimum/lower bound.
    float
        Maximum/upper bound.

    """
    
    mn = min(x)
    mx = max(x)
    d = mx - mn
    return mn - bleed*d, mx + bleed*d

def plot_clf_contours(ax, model, X, y, cmap=plt.cm.coolwarm, 
                      mesh_size = (300,300)):
    """Produces a contour plot of a classifier showing the decision boundaries
    along with the training data. Adds the plot to supplied Axes without showing
    figure so that Axes can be customised.

    Parameters
    ----------
    ax : Axes
        The axes to plot the contours on.
    model : Classifier object with predict() method
        The classifier to visualise.
    X : array-like of shape (n_samples, n_features)
        The design matrix of the training data.
    y : array-like of shape (n_samaples,)
        The class labels for the data.
    cmap : cmap, optional
        The colourmap used for the contours and training data.
        The default is plt.cm.coolwarm.
    mesh_size : (int, int), optional
        The number of points in the (x,y) dimensions of the meshgrid used for 
        drawing the contours. Larger values have finer detail but are slower
        to calculate.

    Returns
    -------
    None.

    """
    # Cast as category just in case
    y = y.astype('category')
    # Generate numeric codes for each category. These will be needed for the
    # creating contours.
    codes = {k:v for (v,k) in enumerate(y.cat.categories)}
    # Convert to array to allow numpy slicing
    X = X.to_numpy()
    # Generate meshgrid
    xx,yy = np.meshgrid(np.linspace(*grid_bounds(X[:,0]),mesh_size[0]),
                        np.linspace(*grid_bounds(X[:,1]),mesh_size[1]))
    # Classify each point in the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Translate into numeric codes
    Z = np.vectorize(codes.get)(Z)
    Z = Z.reshape(xx.shape)
    # Normalisation needed for to ensure point colours match contour colours 
    # for each class.
    norm = plt.Normalize(vmin=min(codes.values()), vmax=max(codes.values()))
    # Plot contours
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.8, norm=norm)
    # Plot training data
    for cat in y.cat.categories:
        ax.scatter(X[y == cat,0],X[y == cat,1],label=cat,
                   color=cmap(norm(codes[cat])), edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.legend()
    return




