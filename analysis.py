#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 19:10:57 2021

@author: JoshM
"""

from data_collection import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def extract_label(df, label_name):
    data = df.drop(columns=[label_name])
    labels = df[label_name]
    return data,labels

def lending_ds():
    df = load_data()
    df = df.drop(columns=['ID','Name','Region','Income','FP.CPI.TOTL.ZG',
                          'EG.USE.ELEC.KH.PC'])
    df = df.dropna()
    return extract_label(df,'Lending')


#TODO: imputation for missing values

#function that takes two variables, performs gridsearch,
#visualises decision boundaries

def plot(df,x_field,y_field,label_field):
    x = df[x_field]
    y = df[y_field]
    labels = df[label_field]
    fig,ax = plt.subplots(dpi=200)
    ax.set_xlabel(x_field)
    ax.set_xscale('log')
    ax.set_ylabel(y_field)
    ax.set_yscale('log')
    for label in labels.unique():
        ax.scatter(x[labels == label],y[labels == label],label=label)
    ax.legend()
    plt.show()
    return

def LDA_2D(X,y):
    #fit LDA and transform data
    lda = LinearDiscriminantAnalysis(n_components = 2)
    lda.fit(X,y)
    print("LDA score: {}".format(lda.score(X,y)))
    X_trans = lda.transform(X)
    
    #plot data
    fig,ax = plt.subplots(dpi=200)
    ax.set_title("LDA plot")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    for label in y.unique():
        ax.scatter(X_trans[y == label,0],X_trans[y == label,1],label=label)
    ax.legend()
    plt.show()
    return X_trans

def linear_SVM_curve(X, y, C_range = np.logspace(-4,4,9)):
    #train classifiers
    model = Pipeline(steps=[('scaler',StandardScaler()),
                            ('svc',LinearSVC())])
    train_scores, val_scores = validation_curve(model,
                                                 X, y, param_name='svc__C',
                                                 param_range=C_range, cv=5)
    #calculate mean and standard deviations #TODO: switch over to grid search
    train_mean = [i.mean() for i in train_scores]
    val_mean = [i.mean() for i in val_scores]
    val_std = [i.std() for i in val_scores]
    #output max
    idx_max = val_mean.index(max(val_mean))
    print("Maximum mean validation score was {0:.3f} at C {1:.3f}".format(
        val_mean[idx_max],C_range[idx_max]))
    
    #plot results
    fig,ax = plt.subplots(dpi=200)
    ax.set_title("Linear SVC comparison")
    ax.set_xlabel("Regularisation coefficient")
    ax.set_ylabel("Accuracy")
    ax.plot(C_range,train_mean,'b-',label="Training score")
    ax.errorbar(C_range,val_mean,val_std,fmt='r-',capsize=3,
                label="Validation Accuracy")
    ax.set_xscale('log')
    ax.legend()
    plt.show()
    return model

def RBF_SVM_grid(X,y,C_range = np.logspace(-1,8,10),
                 gamma_range = np.logspace(-7,2,10)):
    param_grid = {'svc__C':C_range, 'svc__gamma':gamma_range}
    model = Pipeline(steps=[('scaler',StandardScaler()),
                           ('svc',SVC(kernel='rbf'))])
    grid = GridSearchCV(model, param_grid=param_grid,
                        cv=5,return_train_score=True)
    #train classifiers
    grid.fit(X,y)
    #output max
    print("Maximum mean validation score was {0:.3f} with parameters {1}".format(
        grid.best_score_,grid.best_params_))
    
    #plot heatmap
    fig,ax = plt.subplots(dpi=200)
    plot_grid_heatmap(ax,grid,param_grid)
    ax.set_xlabel("Gamma")
    ax.set_ylabel("C")
    ax.set_title("RBF SVC Grid Search")
    plt.show()
    
    #plot slices
    fig,axs = plt.subplots(1,2,dpi=300,figsize=(12,6),sharey=True)
    axs[1].tick_params(labelleft=True)
    plot_grid_slices(axs, grid, param_grid)
    axs[0].set_xlabel("Gamma")
    axs[0].set_xscale('log')
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    axs[1].set_xlabel("C")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_xscale('log')
    axs[1].legend()
    plt.show()
    return

def plot_grid_heatmap(ax, grid, param_grid, cmap = plt.cm.plasma):
    (var1_name, var1_range), (var2_name, var2_range) = param_grid.items()
    scores = grid.cv_results_["mean_test_score"].reshape(len(var1_range),
                                                     len(var2_range))
    plt.imshow(scores,cmap=cmap)
    ax.set_xticks(np.arange(len(var2_range)))
    ax.set_xticklabels(var2_range,rotation=90)
    ax.set_yticks(np.arange(len(var1_range)))
    ax.set_yticklabels(var1_range)
    plt.colorbar(label="Validation Accuracy")
    return

def plot_grid_slices(axs, grid, param_grid):
    (var1_name, var1_range), (var2_name, var2_range) = param_grid.items()
    var1_best = grid.best_params_[var1_name]
    var2_best = grid.best_params_[var2_name]
    var1_idx = np.where(var1_range == var1_best)
    var2_idx = np.where(var2_range == var2_best)
    val_scores = grid.cv_results_["mean_test_score"].reshape(len(var1_range),
                                                     len(var2_range))
    val_stds = grid.cv_results_["std_test_score"].reshape(len(var1_range),
                                                     len(var2_range))
    train_scores = grid.cv_results_["mean_train_score"].reshape(len(var1_range),
                                                     len(var2_range))
    axs[0].set_xticks(var2_range)
    plot_validation_curve(axs[0], var2_range, train_scores[var1_idx,:].flatten(),
                          val_scores[var1_idx,:].flatten(),
                          val_stds[var1_idx,:].flatten())
    
    axs[1].set_xticks(var1_range)
    plot_validation_curve(axs[1], var1_range, train_scores[:,var2_idx].flatten(),
                          val_scores[:,var2_idx].flatten(),
                          val_stds[:,var2_idx].flatten())
    return

def plot_validation_curve(ax, X, train_mean, val_mean, val_std):
    #training curve
    ax.plot(X, train_mean,'b-',label="Training score")
    #validation curbe with 1 standard deviation error bars
    ax.errorbar(X, val_mean, val_std, fmt='r-',capsize=3,
                label="Validation score")
    return

def plot_SVM_boundaries(ax, model,X,y,cmap=plt.cm.coolwarm):
    #labels must be numeric
    X0 = X[:,0]
    X1 = X[:,1]
    xx,yy = np.meshgrid(np.linspace(min(X0)-1,max(X0)+1,300),
                        np.linspace(min(X1)-1,max(X1)+1,300))
    
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
    ax.scatter(X0,X1,c=y,cmap=cmap,edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    return

# def visualise_SVM(model,X,y,cmap=plt.cm.coolwarm):
#     #labels must be numeric
#     X0 = X[:,0]
#     X1 = X[:,1]
#     xx,yy = np.meshgrid(np.linspace(min(X0)-1,max(X0)+1,300),
#                         np.linspace(min(X1)-1,max(X1)+1,300))
    
#     fig,ax = plt.subplots(dpi=200)
#     ax.set_title("SVC Visualisation")
    
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     ax.contourf(xx, yy, Z,cmap=cmap,alpha=0.8)
    
#     ax.scatter(X0,X1,c=y,cmap=cmap,edgecolors='k')
#     ax.set_xlim(xx.min(), xx.max())
#     ax.set_ylim(yy.min(), yy.max())
#     plt.show()
#     return

# def RBF_SVM_grid(X,y,C_range = np.logspace(-1,8,10),
#                  gamma_range = np.logspace(-7,2,10)):
#     param_grid = {'svc__C':C_range, 'svc__gamma':gamma_range}
#     model = Pipeline(steps=[('scaler',StandardScaler()),
#                            ('svc',SVC(kernel='rbf'))])
#     grid = GridSearchCV(model, param_grid=param_grid,
#                         cv=5,return_train_score=True)
#     #train classifiers
#     grid.fit(X,y)
#     #output max
#     print("Maximum mean validation score was {0:.3f} with parameters {1}".format(
#         grid.best_score_,grid.best_params_))
    
#     plot_SVM_grid(grid,param_grid)
#     return


# def plot_SVM_grid(grid, param_grid):
#     C_range = param_grid['svc__C']
#     gamma_range = param_grid['svc__gamma']
#     #plot heatmap of gridsearch
#     scores = grid.cv_results_["mean_test_score"].reshape(len(C_range),
#                                                      len(gamma_range))
#     fig,ax = plt.subplots(dpi=200)
#     plt.imshow(scores,cmap=plt.cm.plasma)
#     ax.set_xlabel("Gamma")
#     ax.set_ylabel("C")
#     ax.set_xticks(np.arange(len(gamma_range)))
#     ax.set_xticklabels(gamma_range,rotation=90)
#     ax.set_yticks(np.arange(len(C_range)))
#     ax.set_yticklabels(C_range)
#     ax.set_title("RBF SVC Grid Search")
#     plt.colorbar(label="Validation Accuracy")
#     plt.show()
    
#     #plot slices of each parameter
#     C_best = grid.best_params_["svc__C"]
#     gamma_best = grid.best_params_["svc__gamma"]
#     C_idx = np.where(C_range == C_best)
#     gamma_idx = np.where(gamma_range == gamma_best)
#     val_stds = grid.cv_results_["std_test_score"].reshape(len(C_range),
#                                                      len(gamma_range))
#     train_scores = grid.cv_results_["mean_train_score"].reshape(len(C_range),
#                                                      len(gamma_range))
#     fig,axs = plt.subplots(1,2,dpi=300,figsize=(12,6))
#     fig.suptitle("RBF SVC Validation Curves")
#     #plot gamma curve
#     axs[0].set_title("Validation Curve for C = {0}".format(C_best))
#     axs[0].set_xlabel("Gamma")
#     axs[0].set_xticks(gamma_range)
#     axs[0].set_ylabel("Accuracy")
#     axs[0].set_xscale('log')
#     axs[0].plot(gamma_range,train_scores[C_idx,:].flatten(),'b-',
#                 label="Training score")
#     axs[0].errorbar(gamma_range,scores[C_idx,:].flatten(),
#                     val_stds[C_idx,:].flatten(),fmt='r-',capsize=3,
#                     label="Validation score")
#     axs[0].legend()
#     #plot C curve
#     axs[1].set_title("Validation Curve for Gamma = {0}".format(gamma_best))
#     axs[1].set_xlabel("C")
#     axs[1].set_xticks(C_range)
#     axs[1].set_ylabel("Accuracy")
#     axs[1].set_xscale('log')
#     axs[1].plot(C_range,train_scores[:,gamma_idx].flatten(),'b-',
#                 label="Training score")
#     axs[1].errorbar(C_range,scores[:,gamma_idx].flatten(),
#                     val_stds[:,gamma_idx].flatten(),fmt='r-',capsize=3,
#                     label="Validation score")
#     axs[1].legend()
#     plt.show()
#     return
