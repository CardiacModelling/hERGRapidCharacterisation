# coding: utf-8
#
# Utility functions for plotting that is not in pints.plot
#
# Contains functions:
#  - plot_covariance()
#  - plot_covariance_trace()
#  - plot_posterior_predictive_distribution()
#  - change_labels_pairwise()
#  - change_labels_trace()
#  - change_labels_histogram()
#
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def plot_covariance(samples,
                    corr=False,
                    kde=False,
                    ref_parameters=None,
                    n_percentiles=None):
    """
    Take a list of covariance matrix samples and creates a distribution plot
    of the covariance matrix.
    
    Idea is to see if this can resemble the covariance matrix that used to 
    create the low-level individual experiments (observations).
    
    `samples`
        A list of samples of covariance matrix, with shape 
        `(n_samples, dimension, dimension)`.
    `corr`
        (Optional) If True, plot correlation matrix instead.
    `kde`
        (Optional) Set to `True` to use kernel-density estimation for the
        histograms and scatter plots.
    `ref_parameters`
        (Optional) A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting.
    `n_percentiles`
        (Optional) Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in `samples`.

    Returns a `matplotlib` figure object and axes handle.
    """
    
    # Check samples size
    try:
        n_sample, n_param, n_param_tmp = samples.shape
    except ValueError:
        raise ValueError('`samples` must be of shape (n_sample, n_param,'
                         ' n_param)')
    
    # Check dimension
    if n_param != n_param_tmp:
        raise ValueError('Covariance matrix must be a square matrix')
    
    # Create figure
    fig_size = (3 * n_param, 3 * n_param)
    fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)
    
    bins = 30
    for i in range(n_param):
        for j in range(n_param):
            if corr and False:
                raise NotImplementedError
                norm = np.sqrt(samples[:, i, i]) * np.sqrt(samples[:, j, j])
            else:
                norm = 1.0
            samples[:, i, j] = samples[:, i, j] / norm

            if i == j and not corr:
                # Diagonal: Plot a histogram
                if n_percentiles is None:
                    xmin = np.min(samples[:, i, i])
                    xmax = np.max(samples[:, i, i])
                else:
                    xmin = np.percentile(samples[:, i, i],
                                         50 - n_percentiles / 2.)
                    xmax = np.percentile(samples[:, i, i],
                                         50 + n_percentiles / 2.)
                xbins = np.linspace(xmin, xmax, bins)
                axes[i, j].set_xlim(xmin, xmax)
                axes[i, j].hist(samples[:, i, i], bins=xbins, normed=True)

                # Add kde plot
                if kde:
                    x = np.linspace(xmin, xmax, 100)
                    axes[i, j].plot(x, stats.gaussian_kde(samples[:, i, i])(x))

                # Add reference parameters if given
                if ref_parameters is not None:
                    ymin_tv, ymax_tv = axes[i, j].get_ylim()
                    axes[i, j].plot(
                        [ref_parameters[i, i], ref_parameters[i, i]],
                        [0.0, ymax_tv],
                        '--', c='k')
            elif i == j  and corr:
                axes[i, j].set_xlim(-1, 1)
                axes[i, j].set_yticklabels([])
            
            elif i < j:
                # Top-right: no plot
                axes[i, j].axis('off')
                
            else:
                # Lower-left: Plot a histogram again
                if n_percentiles is None:
                    xmin = np.min(samples[:, i, j])
                    xmax = np.max(samples[:, i, j])
                else:
                    xmin = np.percentile(samples[:, i, j],
                                         50 - n_percentiles / 2.)
                    xmax = np.percentile(samples[:, i, j],
                                         50 + n_percentiles / 2.)
                xbins = np.linspace(xmin, xmax, bins)
                if corr:
                    axes[i, j].set_xlim(-1, 1)
                else:
                    axes[i, j].set_xlim(xmin, xmax)
                axes[i, j].hist(samples[:, i, j], bins=xbins, normed=True)

                # Add kde plot
                if kde:
                    x = np.linspace(xmin, xmax, 100)
                    axes[i, j].plot(x, stats.gaussian_kde(samples[:, i, j])(x))

                # Add reference parameters if given
                if ref_parameters is not None:
                    ymin_tv, ymax_tv = axes[i, j].get_ylim()
                    axes[i, j].plot(
                        [ref_parameters[i, j], ref_parameters[i, j]],
                        [0.0, ymax_tv],
                        '--', c='k')
                
            # Set tick labels
            if i < n_param - 1 and corr:
                # Only show x tick labels for the last row
                axes[i, j].set_xticklabels([])
            elif corr:
                # Rotate the x tick labels to fit in the plot
                for tl in axes[i, j].get_xticklabels():
                    tl.set_rotation(30)
            
        # Set axis labels
        axes[-1, i].set_xlabel('Parameter %d' % (i + 1))
        axes[i, 0].set_ylabel('Frequency')
        
    return fig, axes


def plot_variable_covariance(sample_mean, sample_cov,
                             ref_parameters=None,
                             n_percentiles=None,
                             fig=None, axes=None, colour=None):
    # TODO
    """
    Take a list of covariance matrix samples and creates a pairwise variable 
    covariance base on the distribution -- assuming multivariate normal for
    now;  
    TODO: might need to do lognormal case too!
    
    Idea is to see if this can resemble the covariance matrix that used to 
    create the low-level individual experiments (observations).
    
    `samples`
        A list of samples of covariance matrix, with shape 
        `(n_samples, dimension, dimension)`.
    `ref_parameters`
        (Optional) A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting. Pass it as [ref_mean, ref_cov].
    `n_percentiles`
        (Optional) Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in `samples`.

    Returns a `matplotlib` figure object and axes handle.
    """
    
    # Check samples size
    try:
        n_sample, n_param, n_param_tmp = sample_cov.shape
    except ValueError:
        raise ValueError('`samples` must be of shape (n_sample, n_param,'
                         ' n_param)')
    
    # Check dimension
    if n_param != n_param_tmp:
        raise ValueError('Covariance matrix must be a square matrix')
    
    # Check mean
    if sample_mean.shape[0] != n_sample:
        print(sample_mean.shape[0], n_sample)
        raise ValueError('Number of input means must match input covariance')
    if sample_mean.shape[1] != n_param:
        raise ValueError('Number of input mean parameters must match input'
                         ' covariance')
    
    # Warning
    if n_sample > 1000:
        print('WARNING: Large sample size might take long to plot')
    
    # Create figure
    if fig is None and axes is None:
        fig_size = (3 * n_param, 3 * n_param)
        fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)

    # Set colour
    if colour is None:
        colour = '#1f77b4'
    
    def normal1D(x, mu, sigma):
        # normal distribution
        output = 1/(sigma * np.sqrt(2 * np.pi)) * \
                 np.exp( - (x - mu)**2 / (2 * sigma**2) )
        return output
    
    for i in range(n_param):
        for j in range(n_param):

            if i == j:
                # Diagonal: Plot a 1D distribution
                
                # 2 sigma covers up 95.5%
                xmin = np.min(sample_mean[:, i]) \
                       - 2.5 * np.max(np.sqrt(sample_cov[:, i, i]))
                xmax = np.max(sample_mean[:, i]) \
                       + 2.5 * np.max(np.sqrt(sample_cov[:, i, i]))
                xx = np.linspace(xmin, xmax, 100)
                axes[i, j].set_xlim(xmin, xmax)
                
                for m, s in zip(sample_mean[:, i], sample_cov[:, i, i]):
                    axes[i, j].plot(xx, normal1D(xx, m, np.sqrt(s)), 
                                   c=colour, alpha=0.2)  #003366

                # Add reference parameters if given
                if ref_parameters is not None:
                    m, s = ref_parameters[0][i], ref_parameters[1][i, i]
                    axes[i, j].plot(xx, normal1D(xx, m, np.sqrt(s)), 
                                    '--', c='r', lw=2)
            
            elif i < j:
                # Top-right: no plot
                axes[i, j].axis('off')
                
            else:
                # Lower-left: Plot a bivariate contour of CI
                
                # 2 sigma covers up 95.5%
                xmin = np.min(sample_mean[:, j]) \
                       - 2.5 * np.max(np.sqrt(sample_cov[:, j, j]))
                xmax = np.max(sample_mean[:, j]) \
                       + 2.5 * np.max(np.sqrt(sample_cov[:, j, j]))
                ymin = np.min(sample_mean[:, i]) \
                       - 2.5 * np.max(np.sqrt(sample_cov[:, i, i]))
                ymax = np.max(sample_mean[:, i]) \
                       + 2.5 * np.max(np.sqrt(sample_cov[:, i, i]))
                axes[i, j].set_xlim(xmin, xmax)
                axes[i, j].set_ylim(ymin, ymax)
                
                for m, s in zip(sample_mean, sample_cov):
                    # for xj, yi
                    mu = np.array([m[j], m[i]])
                    cov = np.array([[ s[j, j], s[j, i] ], 
                                    [ s[i, j], s[i, i] ]])
                    xx, yy = plot_cov_ellipse(mu, cov)
                    axes[i, j].plot(xx, yy, c=colour, alpha=0.2)  #003366

                # Add reference parameters if given
                if ref_parameters is not None:
                    m, s = ref_parameters
                    # for xj, yi
                    mu = np.array([m[j], m[i]])
                    cov = np.array([[ s[j, j], s[j, i] ], 
                                    [ s[i, j], s[i, i] ]])
                    xx, yy = plot_cov_ellipse(mu, cov)
                    axes[i, j].plot(xx, yy, '--', c='r', lw=2)
                
            # Set tick labels
            if i < n_param - 1:
                # Only show x tick labels for the last row
                axes[i, j].set_xticklabels([])
            else:
                # Rotate the x tick labels to fit in the plot
                for tl in axes[i, j].get_xticklabels():
                    tl.set_rotation(30)

            if j > 0 and j != i:
                # Only show y tick labels for the first column and diagonal
                axes[i, j].set_yticklabels([])

        # Set axis labels
        axes[-1, i].set_xlabel('Parameter %d' % (i + 1))
        if i == 0:
            # The first one is not a parameter
            axes[i, 0].set_ylabel('Frequency')
        else:
            axes[i, 0].set_ylabel('Parameter %d' % (i + 1))
        
    return fig, axes


def plot_correlation_and_variable_covariance(sample_mean, sample_cov,
                                             samples, corr=False,
                                             ref_parameters=None,
                                             n_percentiles=None,
                                             fig=None, axes=None, colours=None):
    # TODO
    """
    Take a list of covariance matrix samples and creates a pairwise variable 
    covariance base on the distribution -- assuming multivariate normal for
    now;  
    TODO: might need to do lognormal case too!
    
    Idea is to see if this can resemble the covariance matrix that used to 
    create the low-level individual experiments (observations).
    
    `samples`
        A list of samples of covariance matrix, with shape 
        `(n_samples, dimension, dimension)`.
    `ref_parameters`
        (Optional) A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting. Pass it as [ref_mean, ref_cov].
    `n_percentiles`
        (Optional) Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in `samples`.

    Returns a `matplotlib` figure object and axes handle.
    """
    
    # Check samples size
    try:
        n_sample, n_param, n_param_tmp = sample_cov.shape
    except ValueError:
        raise ValueError('`samples` must be of shape (n_sample, n_param,'
                         ' n_param)')
    
    # Check dimension
    if n_param != n_param_tmp:
        raise ValueError('Covariance matrix must be a square matrix')
    
    # Check mean
    if sample_mean.shape[0] != n_sample:
        print(sample_mean.shape[0], n_sample)
        raise ValueError('Number of input means must match input covariance')
    if sample_mean.shape[1] != n_param:
        raise ValueError('Number of input mean parameters must match input'
                         ' covariance')
    
    # Warning
    if n_sample > 1000:
        print('WARNING: Large sample size might take long to plot')
    
    # Create figure
    if fig is None and axes is None:
        fig_size = (3 * n_param, 3 * n_param)
        fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)

    # Set colour
    if colours is None:
        colours = ['#1f77b4', '#2ca02c', '#ff7f0e']
    else:
        assert(len(colours) == 3)

    bins = 40
    
    def normal1D(x, mu, sigma):
        # normal distribution
        output = 1/(sigma * np.sqrt(2 * np.pi)) * \
                 np.exp( - (x - mu)**2 / (2 * sigma**2) )
        return output
    
    for i in range(n_param):
        for j in range(n_param):

            if i == j:
                # Diagonal: Plot a 1D distribution
                
                # 2 sigma covers up 95.5%
                xmin = np.min(sample_mean[:, i]) \
                       - 2.5 * np.max(np.sqrt(sample_cov[:, i, i]))
                xmax = np.max(sample_mean[:, i]) \
                       + 2.5 * np.max(np.sqrt(sample_cov[:, i, i]))
                xx = np.linspace(xmin, xmax, 100)
                axes[i, j].set_xlim(xmin, xmax)
                
                for m, s in zip(sample_mean[:, i], sample_cov[:, i, i]):
                    axes[i, j].plot(xx, normal1D(xx, m, np.sqrt(s)), 
                                   c=colours[1], alpha=0.2)  #003366
                axes[i, j].tick_params('y', colors=colours[1])

                # Add reference parameters if given
                if ref_parameters is not None:
                    m, s = ref_parameters[0][i], ref_parameters[1][i, i]
                    axes[i, j].plot(xx, normal1D(xx, m, np.sqrt(s)), 
                                    '--', c='k', lw=2.5)
            
            elif i < j:
                if corr and False:
                    raise NotImplementedError
                    norm = np.sqrt(samples[:, i, i]) * np.sqrt(samples[:, j, j])
                else:
                    norm = 1.0
                samples[:, i, j] = samples[:, i, j] / norm
                # Lower-left: Plot a histogram again
                if n_percentiles is None:
                    xmin = np.min(samples[:, i, j])
                    xmax = np.max(samples[:, i, j])
                else:
                    xmin = np.percentile(samples[:, i, j],
                                         50 - n_percentiles / 2.)
                    xmax = np.percentile(samples[:, i, j],
                                         50 + n_percentiles / 2.)
                xbins = np.linspace(xmin, xmax, bins)
                # Set tick and label for y to right
                axes[i, j].yaxis.tick_right()
                axes[i, j].yaxis.set_label_position("right")
                # Only set label for x to top
                axes[i, j].xaxis.tick_top()
                axes[i, j].xaxis.set_label_position("top")
                if corr:
                    axes[i, j].set_xlim(-1, 1)
                else:
                    axes[i, j].set_xlim(xmin, xmax)
                axes[i, j].hist(samples[:, i, j], bins=xbins, normed=True,
                             color=colours[2])

                # Add reference parameters if given
                if ref_parameters is None:
                    ymin_tv, ymax_tv = axes[i, j].get_ylim()
                    axes[i, j].plot(
                        [0, 0],
                        [0.0, ymax_tv],
                        '--', c='k')
                else:
                    if not corr:
                        C = np.copy(ref_parameters[1])
                    else:
                        D = np.sqrt(np.diag(ref_parameters[1]))
                        C = ref_parameters[1] / D / D[:, None]
                    ymin_tv, ymax_tv = axes[i, j].get_ylim()
                    axes[i, j].plot(
                        [C[i, j], C[i, j]],
                        [0.0, ymax_tv],
                        '--', c='k', lw=2.5)

                # Set tick label format
                axes[i, j].ticklabel_format(axis='y', 
                                         style='sci', 
                                         scilimits=(-1,1))
                axes[i, j].tick_params('y', colors=colours[2])
                if i > 0:
                    #axes[i, j].tick_params(axis='x', labelbottom='off',
                    #                    labeltop='off')
                    axes[i, j].set_xticklabels([])
                axes[i, j].tick_params('x', colors=colours[2], labelsize=20)
                axes[i, j].set_xticks([-1.0, 0.0, 1.0])

                # Set x, ylabel
                if i == int(len(axes)/2) and j == int(len(axes)-1):
                    axes[i, j].text(1.25, 0.5, 'Frequency', fontsize=38,
                                    ha='left', va='center', rotation=90,
                                    transform=axes[i, j].transAxes,
                                    color=colours[2])
                elif i == 0 and j == int(len(axes)/2):
                    axes[i, j].text(0.5, 1.25, 'Correlation', fontsize=38,
                                    ha='center', va='bottom',
                                    transform=axes[i, j].transAxes,
                                    color=colours[2])
                
            else:
                # Lower-left: Plot a bivariate contour of CI
                
                # 2 sigma covers up 95.5%
                xmin = np.min(sample_mean[:, j]) \
                       - 2.5 * np.max(np.sqrt(sample_cov[:, j, j]))
                xmax = np.max(sample_mean[:, j]) \
                       + 2.5 * np.max(np.sqrt(sample_cov[:, j, j]))
                ymin = np.min(sample_mean[:, i]) \
                       - 2.5 * np.max(np.sqrt(sample_cov[:, i, i]))
                ymax = np.max(sample_mean[:, i]) \
                       + 2.5 * np.max(np.sqrt(sample_cov[:, i, i]))
                axes[i, j].set_xlim(xmin, xmax)
                axes[i, j].set_ylim(ymin, ymax)
                
                for m, s in zip(sample_mean, sample_cov):
                    # for xj, yi
                    mu = np.array([m[j], m[i]])
                    cov = np.array([[ s[j, j], s[j, i] ], 
                                    [ s[i, j], s[i, i] ]])
                    xx, yy = plot_cov_ellipse(mu, cov)
                    axes[i, j].plot(xx, yy, c=colours[0], alpha=0.2)  #003366

                # Add reference parameters if given
                if ref_parameters is not None:
                    m, s = ref_parameters
                    # for xj, yi
                    mu = np.array([m[j], m[i]])
                    cov = np.array([[ s[j, j], s[j, i] ], 
                                    [ s[i, j], s[i, i] ]])
                    xx, yy = plot_cov_ellipse(mu, cov)
                    axes[i, j].plot(xx, yy, '--', c='k', lw=2.5)
                
            # Set tick labels
            # Only lower triangle
            if i < n_param - 1 and i >= j:
                # Only show x tick labels for the last row
                axes[i, j].set_xticklabels([])
            # else:
            #     # Rotate the x tick labels to fit in the plot
            #     for tl in axes[i, j].get_xticklabels():
            #         tl.set_rotation(30)

            # Only lower triangle
            if j > 0 and j != i and i >= j:
                # Only show y tick labels for the first column and diagonal
                axes[i, j].set_yticklabels([])

        # Set axis labels
        axes[-1, i].set_xlabel('Parameter %d' % (i + 1))
        if i == 0:
            # The first one is not a parameter
            axes[i, 0].set_ylabel('Frequency')
        else:
            axes[i, 0].set_ylabel('Parameter %d' % (i + 1))
        
    return fig, axes


def plot_cov_ellipse(mean, cov, confidence=95):
    """
    Take a 2-D array of mean and 2 x 2 matrix covariance and return two x, y 
    arrays for plotting a ellipse.
    
    `mean`
        A 2-D array of mean
    `cov`
        A 2 x 2 matrix of covariance
    `confidence`
        The confidence interval (in %) specified by the ellipse
    """
    # Calculate the eigenvectors and eigenvalues
    W, V = np.linalg.eig(cov)  # eigenvalues and eigenvectors
    
    # Get the index of the largest eigenvector
    idx_largest_V = np.argmax(W)
    largest_V = V[:, idx_largest_V]
    largest_W = W[idx_largest_V]
    
    # Get the smallest eigenvector and eigenvalue
    smallest_V = V[:, 1 - idx_largest_V]
    smallest_W = W[1 - idx_largest_V]
    
    # Calculate the angle between the x-axis and the largest eigenvector
    phi = np.arctan2(largest_V[1], largest_V[0])
    # Shift angle to between 0 and 2pi
    phi = phi + 2.*np.pi if phi < 0 else phi
    
    # Get the 95% confidence interval error ellipse
    chisquare = 2.4477  # need to work out a conversion from CI to this
    a = chisquare * np.sqrt(largest_W)
    b = chisquare * np.sqrt(smallest_W)
    
    # Ellipse in x and y coordinates 
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x_r = a * np.cos(theta)
    ellipse_y_r = b * np.sin(theta)
    
    # Rotate it
    # Checked it is the right direction http://athenasc.com/Bivariate-Normal.pdf
    xx = np.cos(phi) * ellipse_x_r - np.sin(phi) * ellipse_y_r
    yy = np.sin(phi) * ellipse_x_r + np.cos(phi) * ellipse_y_r
    
    # Rhift it
    xx += mean[0]
    yy += mean[1]
    
    return xx, yy


def plot_covariance_trace(samples,
                          ref_parameters=None,
                          n_percentiles=None):
    """
    Take a list of covariance matrix samples and creates a trace plot of the 
    covariance matrix.
    
    Idea is to see if the list of samples are in steady state.
    
    `samples`
        A list of samples of covariance matrix, with shape 
        `(n_samples, dimension, dimension)`.
    `ref_parameters`
        (Optional) A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting.
    `n_percentiles`
        (Optional) Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in `samples`.

    Returns a `matplotlib` figure object and axes handle.
    """
    return None


def plot_posterior_predictive_distribution(samples, 
                                           exp_samples=None, 
                                           hyper_func=None,
                                           normalise=False,
                                           gx=None,
                                           gx_args=None,
                                           fold=False,
                                           ref_hyper=None,
                                           n_percentiles=None,
                                           mode=None,
                                           fig=None, axes=None, axes2=None):
    """
    Plot marginal distributions of the integrated posterior predictive
    distribution.
    
    If exp_samples is specified, the posterior predictive distribution will be
    plotted on top of the individual distribution of the exp_samples.
    
    `samples'
        A list of samples of the hyperparameters. It should have the shape of
        (n_type_of_hyperparameters, n_samples, n_hyperparameters).
        E.g. [ [list_of_hyper_mean], [list_of_hyper_std] ].
    `exp_samples`
        (Optional) A list of samples of the individual experiments. It should
        have the shape of (n_exp, n_samples, n_params).
        E.g. [ [samples_of_exp_1], [samples_of_exp_2], [samples_of_exp_3] ].
    `hyper_func`
        (Optional) The marginal distribution defined by the hyperparameters.
        The first argument takes the index of variable to be marginalised. The
        second argument should define the region of output (over that 
        variable). And it takes *one* sample of the hyperparameters as the 
        rest of the args. It should return the PDE overall the input region.
        If not specified, assume it is multivariate Guassian.
        Predefined options:
        - `None` or 'normal': normal distribution
        - 'log': log-normal distribution
        - 'transform-normal': `gx` transformed-normal distribution
    `gx`
        (Optional) Required for `hyper_func`='hypercube'. The transformation 
        function from model parameter to hyper unit cube space and its first
        derivative: ( g(x), g'(x) )
    `gx_args`
        (Optional) The arguments for gx, used as gx[i](x, *gx_args[i])
    `fold`
        (Optional) If True, plot 3 x 3 plot
    `ref_hyper`
        (Optional) If specified, [mean, stddev], plot reference distribution.
    """
    if exp_samples is not None:
        '''
        try:
            import pints.plot
        except ImportError:
            raise ImportError('To plot individual experiments\' distribution,'
                              ' module pints.plot is required.')
        '''
        # Plot it!
        # fig, axes = pints.plot.histogram(exp_samples)
        if fold:
            fig, axes = histogram_fold(exp_samples, 
                                       normalise=normalise, 
                                       n_percentiles=n_percentiles,
                                       mode=mode,
                                       fig=fig, axes=axes)
            n_param = 9  # assume Kylie's model...
        else:
            fig, axes = histogram(exp_samples,
                                  normalise=normalise,
                                  n_percentiles=n_percentiles)
            n_param = len(axes)
    else:
        # Create a figure
        n_param = len(samples[0][0])
        fig, axes = plt.subplots(n_param, 1, figsize=(8, n_param*2))
    
    # Turn samples to numpy, make indexing easier!
    samples = np.array(samples)
    
    # Define the marginal distribution defined by the hyperparameters
    if hyper_func == None or hyper_func == 'normal':
        def hyper_func(i, x, sample):
            # normal distribution
            mu = sample[0, i]
            sigma = sample[1, i]
            output = 1/(sigma * np.sqrt(2 * np.pi)) * \
                     np.exp( - (x - mu)**2 / (2 * sigma**2) )
            return output
    elif hyper_func == 'log':
        def hyper_func(i, x, sample):
            # log-normal distribution
            # see my note p.97
            mu = sample[0, i]
            sigma = sample[1, i]
            output = 1/(sigma * np.sqrt(2 * np.pi) * x) * \
                     np.exp( - (np.log(x) - mu)**2 / (2 * sigma**2) )
            return output
    elif hyper_func == 'transform-normal':
        if gx == None:
            raise ValueError('gx is required when hyper_func set to be'
                    ' \'transform-normal\'')
        def hyper_func(i, x, sample, gx=gx):
            # unit hypercube transformed normal distribution
            # see my note p.98
            mu = sample[0, i]  # in y
            sigma = sample[1, i]  # in y
            y = gx[0](x, i, *gx_args[0])
            dydx = gx[1](x, i, *gx_args[1])
            output = 1/(sigma * np.sqrt(2 * np.pi)) * \
                     np.exp( - (y - mu)**2 / (2 * sigma**2) ) * \
                     dydx
            return output
    else:
        if not callable(hyper_func):
            raise ValueError('hyper_func must be callable')    
   
    # Estimate where to plot...
    xmin, xmax = [], []
    if exp_samples is not None and not fold:
        for ax in axes:
            lim = ax.get_xlim()
            xmin.append(lim[0])
            xmax.append(lim[1])
    elif exp_samples is not None and fold:
        for i in range(3):
            for j in range(3):
                lim = axes[i][j].get_xlim()
                xmin.append(lim[0])
                xmax.append(lim[1])
    else:
        mean = np.mean(samples[0, :, :], axis=0)
        std = np.std(samples[1, :, :], axis=0)
        n_std = 3
        for i in range(n_param):
            xmin.append(mean[i] - n_std * std[i])
            xmax.append(mean[i] + n_std * std[i])
    
    if axes2 is None:
        axes2out = []
    else:
        axes2out = axes2
    # Compute marginal distributions of posterior predictive
    # may different from n_param because of noise param
    n_samples = len(samples[0])
    resolution = 250
    for i in range(len(samples[0][0])):  # number of hyper params
        if fold:
            ai, aj = int(i/3), i%3
            if axes2 is None:
                ax_marginal = axes[ai, aj].twinx()
                axes2out.append(ax_marginal)
            else:
                ax_marginal = axes2[i]
        else:
            if axes2 is None:
                ax_marginal = axes[i].twinx()
                axes2out.append(ax_marginal)
            else:
                ax_marginal = axes2[i]
        marginal_ppd_i = np.zeros(resolution)
        x = np.linspace(xmin[i], xmax[i], resolution)
        # integrate marginal distribution
        for t in range(n_samples):  # number of samples
            marginal_ppd_i += hyper_func(i, x, samples[:, t, :])
        # normalise it by number of sum
        marginal_ppd_i = marginal_ppd_i / n_samples
        if mode is None:
            ax_marginal.plot(x, marginal_ppd_i, 
                             lw=2, color='#CC0000',
                             label='Post. pred.')
        elif mode == 2:
            ax_marginal.plot(x, marginal_ppd_i, 
                             lw=2, ls='--', color='k',
                             label='Post. pred.')
        if ref_hyper is not None:
            # plot reference hyper distribution (the one that generate params)
            ref_marginal = hyper_func(i, x, ref_hyper)
            ax_marginal.plot(x, ref_marginal,
                             lw=2, ls='--', color='k', 
                             label='True')
            ax_marginal.legend(loc=2)
        if fold:
            ax_marginal.ticklabel_format(axis='y', 
                                         style='sci', 
                                         scilimits=(-1,1))
            if aj == 2 and ai == 1:
                ax_marginal.set_ylabel('Probability density', 
                                       color='#CC0000',
                                       fontsize=16)
        else:
            ax_marginal.set_ylabel('Probability density', color='#CC0000')
        ax_marginal.tick_params('y', colors='#CC0000')
    
    plt.tight_layout()
    return [fig, axes2out], axes


def plot_posterior_predictive_distribution_2col(samples, 
                                           exp_samples=None, 
                                           hyper_func=None,
                                           normalise=False,
                                           gx=None,
                                           gx_args=None,
                                           fold=False,
                                           ref_hyper=None,
                                           n_percentiles=None):
    """
    Plot marginal distributions of the integrated posterior predictive
    distribution.

    Assume giving 10 parameters and plotting in 2x5 plot. And it will move 1st
    one to the 9th (plotting g with noise).
    
    If exp_samples is specified, the posterior predictive distribution will be
    plotted on top of the individual distribution of the exp_samples.
    
    `samples'
        A list of samples of the hyperparameters. It should have the shape of
        (n_type_of_hyperparameters, n_samples, n_hyperparameters).
        E.g. [ [list_of_hyper_mean], [list_of_hyper_std] ].
    `exp_samples`
        (Optional) A list of samples of the individual experiments. It should
        have the shape of (n_exp, n_samples, n_params).
        E.g. [ [samples_of_exp_1], [samples_of_exp_2], [samples_of_exp_3] ].
    `hyper_func`
        (Optional) The marginal distribution defined by the hyperparameters.
        The first argument takes the index of variable to be marginalised. The
        second argument should define the region of output (over that 
        variable). And it takes *one* sample of the hyperparameters as the 
        rest of the args. It should return the PDE overall the input region.
        If not specified, assume it is multivariate Guassian.
        Predefined options:
        - `None` or 'normal': normal distribution
        - 'log': log-normal distribution
        - 'transform-normal': `gx` transformed-normal distribution
    `gx`
        (Optional) Required for `hyper_func`='hypercube'. The transformation 
        function from model parameter to hyper unit cube space and its first
        derivative: ( g(x), g'(x) )
    `gx_args`
        (Optional) The arguments for gx, used as gx[i](x, *gx_args[i])
    `fold`
        (Optional) If True, plot 3 x 3 plot
    `ref_hyper`
        (Optional) If specified, [mean, stddev], plot reference distribution.
    """
    if exp_samples is not None:
        '''
        try:
            import pints.plot
        except ImportError:
            raise ImportError('To plot individual experiments\' distribution,'
                              ' module pints.plot is required.')
        '''
        # Plot it!
        # fig, axes = pints.plot.histogram(exp_samples)
        if fold:
            fig, axes = histogram_fold_2col(exp_samples, 
                                       normalise=normalise, 
                                       n_percentiles=n_percentiles)
            n_param = 10  # assume Kylie's model...
        else:
            fig, axes = histogram(exp_samples,
                                  normalise=normalise,
                                  n_percentiles=n_percentiles)
            n_param = len(axes)
    else:
        # Create a figure
        n_param = len(samples[0][0])
        fig, axes = plt.subplots(n_param, 1, figsize=(8, n_param*2))
    
    # Turn samples to numpy, make indexing easier!
    samples = np.array(samples)
    
    # Define the marginal distribution defined by the hyperparameters
    if hyper_func == None or hyper_func == 'normal':
        def hyper_func(i, x, sample):
            # normal distribution
            mu = sample[0, i]
            sigma = sample[1, i]
            output = 1/(sigma * np.sqrt(2 * np.pi)) * \
                     np.exp( - (x - mu)**2 / (2 * sigma**2) )
            return output
    elif hyper_func == 'log':
        def hyper_func(i, x, sample):
            # log-normal distribution
            # see my note p.97
            mu = sample[0, i]
            sigma = sample[1, i]
            output = 1/(sigma * np.sqrt(2 * np.pi) * x) * \
                     np.exp( - (np.log(x) - mu)**2 / (2 * sigma**2) )
            return output
    elif hyper_func == 'transform-normal':
        if gx == None:
            raise ValueError('gx is required when hyper_func set to be'
                    ' \'transform-normal\'')
        def hyper_func(i, x, sample, gx=gx):
            # unit hypercube transformed normal distribution
            # see my note p.98
            mu = sample[0, i]  # in y
            sigma = sample[1, i]  # in y
            y = gx[0](x, i, *gx_args[0])
            dydx = gx[1](x, i, *gx_args[1])
            output = 1/(sigma * np.sqrt(2 * np.pi)) * \
                     np.exp( - (y - mu)**2 / (2 * sigma**2) ) * \
                     dydx
            return output
    else:
        if not callable(hyper_func):
            raise ValueError('hyper_func must be callable')    
   
    if ref_hyper is not None:
        def ref_hyper_func(i, x, ref):
            # normal distribution
            mu = ref[0, i]
            sigma = ref[1, i]
            output = 1/(sigma * np.sqrt(2 * np.pi)) * \
                     np.exp( - (x - mu)**2 / (2 * sigma**2) )
            return output

    # Estimate where to plot...
    xmin, xmax = [], []
    if exp_samples is not None and not fold:
        for ax in axes:
            lim = ax.get_xlim()
            xmin.append(lim[0])
            xmax.append(lim[1])
    elif exp_samples is not None and fold:
        for i in range(5):
            for j in range(2):
                lim = axes[i][j].get_xlim()
                xmin.append(lim[0])
                xmax.append(lim[1])
    else:
        mean = np.mean(samples[0, :, :], axis=0)
        std = np.std(samples[1, :, :], axis=0)
        n_std = 3
        for i in range(n_param):
            xmin.append(mean[i] - n_std * std[i])
            xmax.append(mean[i] + n_std * std[i])

    #'''
    # Swap order 1st and 9th
    xmin[-2], xmin[-1] = xmin[-1], xmin[-2]
    xmin = np.roll(xmin, 1)
    xmax[-2], xmax[-1] = xmax[-1], xmax[-2]
    xmax = np.roll(xmax, 1)
    #'''
    
    # Compute marginal distributions of posterior predictive
    # may different from n_param because of noise param
    n_samples = len(samples[0])
    resolution = 250
    for i in range(len(samples[0][0])):  # number of hyper params
        if fold:
            # Swap order 1st and 9th
            ai, aj = int((i - 1)/2), (i - 1) % 2
            if i == 0:
                ai, aj = 4, 0
            ax_marginal = axes[ai, aj].twinx()
        else:
            ax_marginal = axes[i].twinx()
        marginal_ppd_i = np.zeros(resolution)
        x = np.linspace(xmin[i], xmax[i], resolution)
        # integrate marginal distribution
        for t in range(n_samples):  # number of samples
            marginal_ppd_i += hyper_func(i, x, samples[:, t, :])
        # normalise it by number of sum
        marginal_ppd_i = marginal_ppd_i / n_samples
        ax_marginal.plot(x, marginal_ppd_i, 
                         lw=2, color='#CC0000',
                         label='Post. pred.')
        if ref_hyper is not None:
            # plot reference hyper distribution (the one that generate params)
            ref_marginal = ref_hyper_func(i, x, ref_hyper)
            ax_marginal.plot(x, ref_marginal,
                             lw=2, ls='--', color='k', 
                             label='True')
            ax_marginal.legend(loc=2)
        if fold:
            ax_marginal.ticklabel_format(axis='y', 
                                         style='sci', 
                                         scilimits=(-1,1))
            if aj == 1 and ai == 2:
                ax_marginal.set_ylabel('Probability\ndensity', 
                                       color='#CC0000',
                                       fontsize=16)
        else:
            ax_marginal.set_ylabel('Probability density', color='#CC0000')
        ax_marginal.tick_params('y', colors='#CC0000')
    
    plt.tight_layout()
    return fig, axes


def histogram(samples,
              ref_parameters=None,
              n_percentiles=None, 
              normalise=False):
    """
    Takes one or more markov chains or lists of samples as input and creates
    and returns a plot showing histograms for each chain or list of samples.
    Arguments:
    ``samples``
        A list of lists of samples, with shape
        ``(n_lists, n_samples, dimension)``, where ``n_lists`` is the number of
        lists of samples, ``n_samples`` is the number of samples in one list \
        and ``dimension`` is the number of parameters.
    ``ref_parameters``
        (Optional) A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting.
    ``n_percentiles``
        (Optional) Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in ``samples``.
    ``normalise``
        (Optional) Shows histograms as probability density functions, i.e., 
        the area (or integral) under the histograms sums to 1.
    Returns a ``matplotlib`` figure object and axes handle.
    """
    # If we switch to Python3 exclusively, bins and alpha can be keyword-only
    # arguments
    bins = 40
    alpha = 0.5
    n_list = len(samples)
    _, n_param = samples[0].shape

    # Check number of parameters
    for samples_j in samples:
        if n_param != samples_j.shape[1]:
            raise ValueError(
                'All samples must have the same number of parameters'
            )

    # Check reference parameters
    if ref_parameters is not None:
        if len(ref_parameters) != n_param:
            raise ValueError(
                'Length of `ref_parameters` must be same as number of'
                ' parameters')

    # Set up figure, plot first samples
    fig, axes = plt.subplots(n_param, 1, figsize=(6, 2 * n_param))
    for i in range(n_param):
        for j_list, samples_j in enumerate(samples):
            # Add histogram subplot
            axes[i].set_xlabel('Parameter ' + str(i + 1))
            if normalise:
                axes[i].set_ylabel('Normalised frequency')
            else:
                axes[i].set_ylabel('Frequency')
            if n_percentiles is None:
                xmin = np.min(samples_j[:, i])
                xmax = np.max(samples_j[:, i])
            else:
                xmin = np.percentile(samples_j[:, i],
                                     50 - n_percentiles / 2.)
                xmax = np.percentile(samples_j[:, i],
                                     50 + n_percentiles / 2.)
            xbins = np.linspace(xmin, xmax, bins)
            axes[i].hist(samples_j[:, i], bins=xbins, alpha=alpha, 
                         density=normalise,
                         label='Samples ' + str(1 + j_list))

        # Add reference parameters if given
        if ref_parameters is not None:
            # For histogram subplot
            ymin_tv, ymax_tv = axes[i].get_ylim()
            axes[i].plot(
                [ref_parameters[i], ref_parameters[i]],
                [0.0, ymax_tv],
                '--', c='k')

    return fig, axes


def change_labels_pairwise(axes, names):
    # Change axes' label to input `names`
    for i in range(len(axes)):
        axes[i][0].set_ylabel(names[i])
        axes[-1][i].set_xlabel(names[i])
    return axes


def change_labels_correlation(axes, names):
    # Change axes' label to input `names`
    for i in range(len(axes)):
        if i != int(len(axes)/2):
            axes[i][0].set_ylabel('')
            axes[-1][i].set_xlabel('')
        else:
            axes[i][0].set_ylabel('Frequency', fontsize=26)
            axes[-1][i].set_xlabel('Correlation', fontsize=26)
        for j in range(len(axes)):
            if i > j:
                if i < len(axes)-1:
                    axes[i][j].tick_params(axis='x', labelbottom='off')
                axes[i][j].ticklabel_format(axis='y', 
                                            style='sci', 
                                            scilimits=(-1,1))
                axes[i][j].set_title(names[i] + '-' + names[j], 
                                     loc='right', fontsize=20)
            if i == j:
                axes[i][j].tick_params(labelcolor='w', 
                                       top='off', bottom='off',
                                       left='off', right='off')
                axes[i][j].tick_params(axis='x', labelbottom='off')
                axes[i][j].tick_params(axis='y', labelleft='off')
    return axes


def change_labels_variable_covariance(axes, names):
    # Change axes' label to input `names`
    for i in range(len(axes)):
        axes[i][0].set_ylabel(names[i], fontsize=26)
        axes[-1][i].set_xlabel(names[i], fontsize=26)
        axes[i][0].ticklabel_format(axis='y', 
                                    style='sci', 
                                    scilimits=(-1,1))
        axes[i][i].ticklabel_format(axis='y', 
                                    style='sci', 
                                    scilimits=(-1,1))
    axes[0][0].set_ylabel('Probability density', fontsize=16)
    return axes


def change_labels_correlation_and_variable_covariance(axes, names):
    # Change axes' label to input `names`
    for i in range(len(axes)):
        axes[i][0].set_ylabel(names[i], fontsize=38)
        if i != 0:
            axes[i][0].tick_params('y', labelsize=20)
        axes[-1][i].set_xlabel(names[i], fontsize=38)
        axes[-1][i].tick_params('x', labelsize=20)
        axes[i][0].ticklabel_format(axis='y', 
                                    style='sci', 
                                    scilimits=(-1,1))
        axes[i][i].ticklabel_format(axis='y', 
                                    style='sci', 
                                    scilimits=(-1,1))
        '''
        for j in range(len(axes)):
            if i < j:
                axes[i+1][j].set_title(names[i] + '-' + names[j], 
                                       loc='center', fontsize=30,
                                       color='#ff7f0e')
        '''
    axes[0][0].set_ylabel('Probability\ndensity', fontsize=32,
                          color='#2ca02c')
    return axes


def change_labels_trace(axes, names):
    # Change axes' label to input `names`
    for i in range(len(axes)):
        axes[i][0].set_xlabel(names[i])
        axes[i][1].set_ylabel(names[i])
    return axes


def change_labels_trace_fold(axes, names):
    # Change axes' label to input `names`
    for i in range(len(axes)):
        for j in range(2):
            axes[i][j*2].set_xlabel(names[i*2 + j], fontsize=18)
            axes[i][j*2+1].set_ylabel(names[i*2 + j], fontsize=18)
    return axes


def change_labels_histogram(axes, names):
    # Change axes' label to input `names`
    for i, ax in enumerate(axes):
        ax.set_xlabel(names[i])
    return axes


def change_labels_histogram_fold(axes, names):
    # Change axes' label to input `names`
    # assume 3x3 and goes to left first.
    for i in range(3):
        for j in range(3):
            axes[i, j].set_xlabel(names[i*3 + j], fontsize=16)
    return axes


def change_labels_histogram_fold_2col(axes, names):
    # Change axes' label to input `names`
    # assume 3x3 and goes to left first.
    for i in range(5):
        for j in range(2):
            axes[i, j].set_xlabel(names[i*2 + j], fontsize=16)
    return axes


def inv_unit_hypercube_to_param_intervals(theta, i, bound):
    an_array = np.copy(theta)
    #for i in range(n):
    #    an_array[i] = np.log(theta[i] - bound[0,i] + 1) \
    #                / np.log(bound[1,i] - bound[0,i] +1)
    an_array = np.log(theta - bound[0,i] + 1) / np.log(bound[1,i] - bound[0,i] +1)
    return an_array


def derivative_inv_unit_hypercube_to_param_intervals(theta, i, bound):
    return 1. / ( np.log(bound[1,i] - bound[0,i] + 1) * (theta - bound[0,i] + 1) )


def histogram_fold(samples,
                   ref_parameters=None,
                   n_percentiles=None, 
                   normalise=False,
                   mode=None,
                   fig=None, axes=None):
    """
    Assume giving 10 parameters and plotting first 9 parameters in a 3x3 plot.
    
    Takes one or more markov chains or lists of samples as input and creates
    and returns a plot showing histograms for each chain or list of samples.
    Arguments:
    ``samples``
        A list of lists of samples, with shape
        ``(n_lists, n_samples, dimension)``, where ``n_lists`` is the number of
        lists of samples, ``n_samples`` is the number of samples in one list \
        and ``dimension`` is the number of parameters.
    ``ref_parameters``
        (Optional) A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting.
    ``n_percentiles``
        (Optional) Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in ``samples``.
    ``normalise``
        (Optional) Shows histograms as probability density functions, i.e., 
        the area (or integral) under the histograms sums to 1.
    Returns a ``matplotlib`` figure object and axes handle.
    """
    # If we switch to Python3 exclusively, bins and alpha can be keyword-only
    # arguments
    bins = 40
    alpha = 0.5
    n_list = len(samples)
    _, n_param = samples[0].shape
    
    if n_param != 10:
        raise ValueError('Assume giving 10 parameters and plotting first 9'
                         ' parameters in a 3x3 plot.')
    n_param -= 1

    # Check number of parameters
    for samples_j in samples:
        if n_param != samples_j.shape[1]-1:
            raise ValueError(
                'All samples must have the same number of parameters'
            )

    # Check reference parameters
    if ref_parameters is not None:
        if len(ref_parameters) != n_param:
            raise ValueError(
                'Length of `ref_parameters` must be same as number of'
                ' parameters')

    # Set up figure, plot first samples
    color_list = ['#1f77b4',
                '#ff7f0e',
                '#2ca02c',
                '#d62728',
                '#9467bd',
                '#8c564b',
                '#e377c2',
                '#7f7f7f',
                '#bcbd22',
                '#17becf',]
    if fig is None or axes is None:
        fig, axes = plt.subplots(3, 3, figsize=(12, 7))
    histtype = 'stepfilled' if mode is None else 'step'
    for i in range(n_param):
        ai, aj = int(i/3), i%3
        for j_list, samples_j in enumerate(samples):
            # Add histogram subplot
            axes[ai, aj].set_xlabel('Parameter ' + str(i + 1))
            if normalise:
                axes[1, 0].set_ylabel('Normalised\nfrequency', fontsize=14)
            else:
                axes[1, 0].set_ylabel('Frequency', fontsize=16)
            if n_percentiles is None:
                xmin = np.min(samples_j[:, i])
                xmax = np.max(samples_j[:, i])
            else:
                xmin = np.percentile(samples_j[:, i],
                                     50 - n_percentiles / 2.)
                xmax = np.percentile(samples_j[:, i],
                                     50 + n_percentiles / 2.)
            xbins = np.linspace(xmin, xmax, bins)
            color = color_list[j_list % len(color_list)]
            axes[ai, aj].hist(samples_j[:, i], bins=xbins, alpha=alpha, color=color,
                         density=normalise, linewidth=1, histtype=histtype,
                         edgecolor=color, label='Samples ' + str(1 + j_list))

        # Add reference parameters if given
        if ref_parameters is not None:
            # For histogram subplot
            ymin_tv, ymax_tv = axes[ai, aj].get_ylim()
            axes[ai, aj].plot(
                [ref_parameters[i], ref_parameters[i]],
                [0.0, ymax_tv],
                '--', c='k')
        axes[ai, aj].ticklabel_format(axis='y', style='sci', scilimits=(-1,1))

    return fig, axes


def histogram_fold_2col(samples,
                   ref_parameters=None,
                   n_percentiles=None, 
                   normalise=False):
    """
    Assume giving 10 parameters and plotting in 2x5 plot. And it will move 1st
    one to the 9th (plotting g with noise).
    
    Takes one or more markov chains or lists of samples as input and creates
    and returns a plot showing histograms for each chain or list of samples.
    Arguments:
    ``samples``
        A list of lists of samples, with shape
        ``(n_lists, n_samples, dimension)``, where ``n_lists`` is the number of
        lists of samples, ``n_samples`` is the number of samples in one list \
        and ``dimension`` is the number of parameters.
    ``ref_parameters``
        (Optional) A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting.
    ``n_percentiles``
        (Optional) Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in ``samples``.
    ``normalise``
        (Optional) Shows histograms as probability density functions, i.e., 
        the area (or integral) under the histograms sums to 1.
    Returns a ``matplotlib`` figure object and axes handle.
    """
    # If we switch to Python3 exclusively, bins and alpha can be keyword-only
    # arguments
    bins = 40
    alpha = 0.5
    n_list = len(samples)
    _, n_param = samples[0].shape
    
    if n_param != 10:
        raise ValueError('Assume giving 10 parameters and plotting first 9'
                         ' parameters in a 3x3 plot.')

    # Check number of parameters
    for samples_j in samples:
        if n_param != samples_j.shape[1]:
            raise ValueError(
                'All samples must have the same number of parameters'
            )
        '''
        # Swap order of plotting: 1st one to the 9th (plotting g with noise)
        samples_j = np.roll(samples_j, -1)
        samples_j[-2], samples_j[-1] = samples_j[-1], samples_j[-2]
        '''

    # Check reference parameters
    if ref_parameters is not None:
        if len(ref_parameters) != n_param:
            raise ValueError(
                'Length of `ref_parameters` must be same as number of'
                ' parameters')

    # Set up figure, plot first samples
    color_list = ['#1f77b4',
                '#ff7f0e',
                '#2ca02c',
                '#d62728',
                '#9467bd',
                '#8c564b',
                '#e377c2',
                '#7f7f7f',
                '#bcbd22',
                '#17becf',]
    fig, axes = plt.subplots(5, 2, figsize=(8, 10))
    for i in range(n_param):
        # ai, aj = int(i/2), i%2
        # Swap order 1st and 9th
        ai, aj = int((i - 1)/2), (i - 1) % 2
        if i == 0:
            ai, aj = 4, 0
        if i == 9:
            ai, aj = 4, 1
        for j_list, samples_j in enumerate(samples):
            # Add histogram subplot
            axes[ai, aj].set_xlabel('Parameter ' + str(i + 1))
            if normalise:
                axes[2, 0].set_ylabel('Normalised\nfrequency', fontsize=14)
            else:
                axes[2, 0].set_ylabel('Frequency', fontsize=16)
            if n_percentiles is None:
                xmin = np.min(samples_j[:, i])
                xmax = np.max(samples_j[:, i])
            else:
                xmin = np.percentile(samples_j[:, i],
                                     50 - n_percentiles / 2.)
                xmax = np.percentile(samples_j[:, i],
                                     50 + n_percentiles / 2.)
            xbins = np.linspace(xmin, xmax, bins)
            color = color_list[j_list % len(color_list)]
            axes[ai, aj].hist(samples_j[:, i], bins=xbins, alpha=alpha, color=color,
                         density=normalise, linewidth=1, histtype='stepfilled',
                         edgecolor=color, label='Samples ' + str(1 + j_list))

        # Add reference parameters if given
        if ref_parameters is not None:
            # For histogram subplot
            ymin_tv, ymax_tv = axes[ai, aj].get_ylim()
            axes[ai, aj].plot(
                [ref_parameters[i], ref_parameters[i]],
                [0.0, ymax_tv],
                '--', c='k')
        axes[ai, aj].ticklabel_format(axis='y', style='sci', scilimits=(-1,1))

    return fig, axes


def trace_fold(samples, ref_parameters=None, n_percentiles=None):
    """
    Assume giving 10 parameters and plotting 5x2(4) plot. 
    
    Takes one or more markov chains or lists of samples as input and creates
    and returns a plot showing histograms and traces for each chain or list of
    samples.

    Arguments:

    `samples`
        A list of lists of samples, with shape
        `(n_lists, n_samples, dimension)`, where `n_lists` is the number of
        lists of samples, `n_samples` is the number of samples in one list and
        `dimension` is the number of parameters.
    `ref_parameters`
        (Optional) A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting.
    `n_percentiles`
        (Optional) Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in `samples`.

    Returns a `matplotlib` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    # If we switch to Python3 exclusively, bins and alpha can be keyword-only
    # arguments
    bins = 40
    alpha = 0.5
    n_list = len(samples)
    _, n_param = samples[0].shape

    # Check number of parameters
    for samples_j in samples:
        if n_param != samples_j.shape[1]:
            raise ValueError(
                'All samples must have the same number of parameters'
            )
    
    if n_param%2 != 0:
        raise ValueError('Assume giving 10 parameters and plotting first 9'
                         ' parameters in a 3x3 plot.')

    # Check reference parameters
    if ref_parameters is not None:
        if len(ref_parameters) != n_param:
            raise ValueError(
                'Length of `ref_parameters` must be same as number of'
                ' parameters')

    # Set up figure, plot first samples
    fig, axes = plt.subplots(int(n_param/2), 4, figsize=(20, n_param/1.25))
    for i in range(n_param):
        ai, aj = int(i / 2), (i%2) * 2
        ymin_all, ymax_all = np.inf, -np.inf
        for j_list, samples_j in enumerate(samples):
            # Add histogram subplot
            if ai == int(n_param / 2 / 2):
                axes[ai, aj].set_ylabel('Frequency', fontsize=18)
            if n_percentiles is None:
                xmin = np.min(samples_j[:, i])
                xmax = np.max(samples_j[:, i])
            else:
                xmin = np.percentile(samples_j[:, i],
                                     50 - n_percentiles / 2.)
                xmax = np.percentile(samples_j[:, i],
                                     50 + n_percentiles / 2.)
            xbins = np.linspace(xmin, xmax, bins)
            axes[ai, aj].hist(samples_j[:, i], bins=xbins, alpha=alpha,
                            histtype='step',
                            label='Chain ' + str(1 + j_list))
            # Set tick label
            axes[ai, aj].ticklabel_format(axis='x', 
                                          style='sci', 
                                          scilimits=(-2,3))
            axes[ai, aj].ticklabel_format(axis='y', 
                                          style='sci', 
                                          scilimits=(-2,3))

            # Add trace subplot
            if ai == int(n_param / 2) - 1:
                axes[ai, aj+1].set_xlabel('Iteration', fontsize=20)
            else:
                axes[ai, aj+1].set_xticklabels([])
            axes[ai, aj+1].plot(samples_j[:, i], alpha=alpha)
            # Set tick label
            axes[ai, aj+1].ticklabel_format(axis='y', 
                                            style='sci', 
                                            scilimits=(-2,3))

            # Set ylim
            ymin_all = ymin_all if ymin_all < xmin else xmin
            ymax_all = ymax_all if ymax_all > xmax else xmax
        axes[ai, aj+1].set_ylim([ymin_all, ymax_all])

        # Add reference parameters if given
        if ref_parameters is not None:
            # For histogram subplot
            ymin_tv, ymax_tv = axes[i, 0].get_ylim()
            axes[ai, aj].plot(
                [ref_parameters[i], ref_parameters[i]],
                [0.0, ymax_tv],
                '--', c='k')

            # For trace subplot
            xmin_tv, xmax_tv = axes[i, 1].get_xlim()
            axes[ai, aj+1].plot(
                [0.0, xmax_tv],
                [ref_parameters[i], ref_parameters[i]],
                '--', c='k')
    if n_list > 1:
        axes[0, 0].legend(loc=4)

    plt.tight_layout()
    return fig, axes



def addbox(axes, subplot, color='silver', alpha=1.0):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    i, j = subplot
    
    # Get boundaries from axes[0, 0]
    # assume all other subplots have the same spacing
    inv = axes[0, 0].transAxes.inverted()
    a = inv.transform(axes[0, 1].transAxes.transform((0, 0)))
    b = inv.transform(axes[1, 0].transAxes.transform((0, 1)))
    a = (a[0] - 1) / 2.  # half hortrizonal dist. between subplots
    b = (-1 * b[1]) / 2.  # half vertical dist. between subplots

    # Vertices
    vertices = [(1, 0),
                (1, 1 + b),
                (1 + a, 1 + b),
                (1 + a, -b),
                (-a, -b),
                (-a, 1 + b),
                (1, 1 + b),
                (1, 1),
                (0, 1),
                (0, 0)]  # surrounding box coordinates in transAxes

    # Add ending
    vertices = list(vertices) + [(0, 0)]

    # Codes
    codes = [Path.MOVETO] \
            + [Path.LINETO] * (len(vertices) - 2) \
            + [Path.CLOSEPOLY]

    pathpatch = PathPatch(Path(vertices, codes),
            facecolor=color,
            linewidth=0,
            clip_on=False,
            transform=axes[i, j].transAxes,
            alpha=alpha,
            zorder=-100)

    # Draw from axes[0, 0]...
    axes[0, 0].add_patch(pathpatch) 


