import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
          
def check_frame(frame, numericals=[], drop = 0, fill=False):

    '''Check/clean a dataframe by identifying missing values and checking numeric column data types. Feature names are converted to snake case
    Numericals - List: Feature names which should contain only numerical values. Prints column name and and any inconsistencies found. Default = False, does not check types
    Drop - Float: Null rows in a feature are dropped if the percentage of null rows is less than threshold. Default = 0, no rows are removed from the data set. Set to 1 if you want all null rows dropped
    Fill - Any: Replace all null values with the passed value. Default = False, leave null values as is'''

    frame.columns = [col.lower().replace(' ','_') for col in frame.columns]
    numericals = [x.lower() for x in numericals]

    if frame.isna().sum().sum() == 0:
        print('No null values in data set')        
    elif fill:
        frame.fillna(fill, inplace=True)
        print(f'All null values converted to {fill}')
    else:
        for col in frame.columns:
            nulls = frame[[col]].isna().sum()[0]
            null_warning = f'{col}: {nulls} null'
            if nulls > 0:
                if (nulls/len(frame)) < drop:
                    frame.dropna(subset=[col], inplace=True)
                    null_warning += ' rows dropped'
                else:
                    null_warning += ' values'
                print(null_warning)

    if len(numericals) > 0:
        print(f'Checking for numerical type mismatches in features: {numericals}')
        no_mismatch = True
        non_numeric = [x for x in numericals if not np.issubdtype(frame[x].dtype, np.number)]
        for non in non_numeric:
            nonval = [y for y in frame[non] if not np.issubdtype(type(y), np.number)]
            print(f'Feature [{non}]: {len(nonval)} non-numeric values')
            no_mismatch = False          

        if no_mismatch:
            print(f'No type mismatches found')

def impute_frame(frame, reference, threshold = 1, modes = [], medians = [], zeroes=[]):

    '''Generalized function for imputing values in a dataframe
    Reference - Dataframe: Training data to take the modes and medians for feature values from
    Threshold - Float: Only impute the column if the percentage of missing values is lower than the threshold. Default = 1: Impute all values
    Modes - List: Feature names that are either categorical or ordinal. Replace null values with the mode of the feature
    Medians - List: Feature names that are either continuous or discrete numerical values. Replace null values with the median of the feature
    Zeroes - List: Feature names that will have their null values replaced by zeroes'''

    for arg_lists in [modes, medians, zeroes]:
        arg_lists = [x.lower().replace(' ','_') for x in arg_lists]

    for mode in modes:
        nulls = frame[mode].isna().sum()
        if nulls/len(frame) < threshold and nulls != 0:                   
            mode_val = statistics.mode(reference[mode].copy().dropna())
            frame.fillna({mode:mode_val}, inplace=True)
            print(f'Feature [{mode}]: {nulls} null values converted to {mode_val}')

    for median in medians:
        nulls = frame[median].isna().sum()
        if nulls/len(frame) < threshold and nulls != 0:
            median_val = statistics.median(reference[median].copy().dropna())
            frame.fillna({median:median_val}, inplace=True)
            print(f'Feature [{median}]: {nulls} null values converted to {median_val}')

    for zero in zeroes:
        nulls = frame[zero].isna().sum()
        if nulls/len(frame) < threshold and nulls != 0:
            frame.fillna({zero:0}, inplace=True)
            print(f'Feature [{zero}]: {nulls} null values converted to 0')

def scale_frame(frame, scalable, scaler):

    '''Normalizes the values of the passed features in the data set
    Scalable - List: Feature names of the data set to be scaled
    Reference - Fitted StandardScaler Instance: Already fitted (to train data) scaler'''

    scaled = scaler.transform(frame[scalable])
    for _ in enumerate(scalable):
        frame.loc[:, _[1]] = scaled.transpose()[_[0]]


def check_range(frame, numericals, correct=False):

    '''Checks the values of numerical features of a dataframe to see if they're within a certain range
    Numericals - Dictionary: Keys are feature names, values are list of minimum/maximum e.g. "Survey Result": [0,5]. A string of 'none' for max/min does not check for that boundary
    Correct: Default - False, leave values in place, but print warning. If set to True, out of bounds values are changed to the minimum or maximum'''

    print(f'Checking for out of range values in features: {list(numericals.keys())}')
    no_errors = True

    for key in numericals:
        key = key.lower().replace(' ','_')
        actual_min, actual_max = frame.describe()[key][3], frame.describe()[key][-1]

        minimum, maximum = numericals[key][0], numericals[key][1]

        if minimum != 'none':
            if actual_min < minimum:
                min_message = ' out of range'
                if correct:
                    frame[key] = frame[key].map(lambda _: minimum if _ < minimum else _)
                    min_message = 's corrected'

                print(f'Feature [{key}] minimum value' + min_message)
                no_errors = False

        if maximum != 'none':
            if actual_max > maximum:
                max_message = ' out of range'
                if correct:
                    frame[key] = frame[key].map(lambda _: maximum if _ < maximum else _)
                    max_message = 's corrected'

                print(f'Feature [{key}] maximum value' + min_message)
                no_errors = False

    if no_errors:
        print(f'No out of range values in selected features')


def match_columns(frame_one, frame_two):

    '''Compares the column headers of two dataframes and creates any missing columns in both frames from the union of the two
    Checks to see if columns are no longer aligned, due to missing values in categorical/ordinal variables. Code taken from stack overflow
    https://stackoverflow.com/questions/28444561/get-only-unique-elements-from-two-lists-python/45098345
    Sort and order columns to match between test and training dataframes'''

    one = frame_one.columns
    two = frame_two.columns

    unique = list(set(one).symmetric_difference(set(two)))
    for uniqcol in unique:
        if uniqcol in one:
            frame_two[uniqcol] = 0
        else:
            frame_one[uniqcol] = 0

    sorted_columns = list(frame_one.columns)
    sorted_columns.sort()

    return frame_one[sorted_columns], frame_two[sorted_columns]

# Returns a reversed slice of a range of indices from a list

def rev_range(x, low=0, high=10):
    return x[low:high][:(low-high)-1:-1]

def collapse_ordinal(frame, key, correct_range, bins):

    '''Collapses an ordinal variable into a smaller number of values and rewrites the original dataframe column with the adjusted numbers. Returned values start with 1
    Key - String: Feature name of the dataframe to collapse
    Correct Range - 2-element List: Contains the actual values for the [minimum, maximum] of the variable
    Bins - Int: Number of values to collapse the ordinal to'''

    def correct_value(bin_ranges, value):
        numbers = np.arange(1, len(bin_ranges))
        for x in range(len(bin_ranges)-1):
            if bin_ranges[x] <= value < bin_ranges[x+1]:
                return numbers[x]
        return numbers[-1]

    bin_list = []
    for x in range(bins):
        bin_list.append(correct_range[0] + x*((correct_range[1] - correct_range[0])/bins))
    bin_list.append(correct_range[1])

    frame[key] = frame[key].map(lambda _: correct_value(bin_list, _))

def create_scatter(x, y, title = None, xlabel = None, ylabel = None, size = (6,4), fitline = []):

    '''Creates a scatter plot using matplotlib and the passed arguments
    x, y - List: x and y coordinate data. Lists must be of equal length
    Title, xlabel, ylabel - Str: Text for the figure title and axes
    Size - Tuple: Two element tuple defining the width and height of the figure
    Fitline - List: Two element list to draw a custom fit line on the chart. List should include two nested lists of two elements start [x,y] and finish [x,y] coordinates for the line'''

    mpl.style.use('ggplot')

    plt.figure(figsize=size)
    plt.scatter(x, y)
    if len(fitline) != 0:
        plt.plot(fitline[0], fitline[1], color='blue', linewidth=2)

    xmax, xmin = plt.xticks()[0][1], plt.xticks()[0][-2]
    ymax, ymin = plt.yticks()[0][1], plt.yticks()[0][-2]
    plt.hlines(0, xmin, xmax, color='k', linewidth=2)
    plt.vlines(0, ymin, ymax, color='k', linewidth=2)
    plt.grid(False)

    plt.xlabel(xlabel, fontsize=10)
    plt.xticks(fontsize=8)
    plt.ylabel(ylabel, fontsize=10)
    plt.yticks(fontsize=8)
    plt.title(title, fontsize=14)
    plt.show()        

def create_hist(x, title = None, xlabel = None, ylabel = None, size = (6,4), bins = 10):

    '''Create a histogram using matplotlib and the passed arguments
    x - List: Series data to be plotted
    Title, xlabel, ylabel - Str: Text for the figure title and axes
    Size - Tuple: Two element tuple defining the width and height of the figure
    Bins - Int: Number of bins to assign values to. Default = 10'''

    mpl.style.use('ggplot')

    plt.figure(figsize=size)
    plt.hist(x, bins=bins)

    plt.xlabel(xlabel, fontsize=10)
    plt.xticks(fontsize=8)
    plt.ylabel(ylabel, fontsize=10)
    plt.yticks(fontsize=8)
    plt.title(title, fontsize=14)
    plt.show()
    return plt

def create_bar(x, title = None, xlabel = None, ylabel = None, ticks = '', size = (6,4), axis = 'v'):

    '''Create a bar chart using matplotlib and the passed arguments
    x - List: Series data to be plotted
    Title, xlabel, ylabel - Str: Text for the figure title and axes. Axes are reversed for horionztal charts
    Size - Tuple: Two element tuple defining the width and height of the figure
    Axis - Str: h for horizontal bar charts, y for vertical. Default = vertical
    Code for adding bar labels taken from stack overflow
    https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh'''

    mpl.style.use('ggplot')

    plt.figure(figsize=size)
    if axis == 'h':
        plt.barh(np.arange(1, len(x)+1), x, tick_label = ticks)
        plt.xlabel(ylabel, fontsize=10)
        plt.ylabel(xlabel, fontsize=10)

        for i, v in enumerate(x):
            plt.text(v + 0.02, i + .9, str(v), color='blue', fontweight='bold')
    else:
        plt.bar(np.arange(1, len(x)+1), x, tick_label = ticks)
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)

    plt.title(title, fontsize=14)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.show()

def create_box(x, title = None, label = None, size = (6,4)):

    mpl.style.use('ggplot')

    plt.figure(figsize=size)
    plt.boxplot(x, vert=False, widths=0.6)

    plt.yticks([])

    plt.xlabel(label, fontsize=10)
    plt.xticks(fontsize=8)
    plt.title(title, fontsize=14)
    plt.show()
    
def subplot(type_list, x_list, title_list, xlab_list, ylab_list, y_list = None, fig_title='', h_size='default', bar_numbers=False):
    
    mpl.style.use('ggplot')
    sup_y = 1
    if len(type_list) == 2:
        columns = 2
    else:
        columns = 3
    
    nrows = int(np.ceil(len(x_list)/3))
    if h_size == 'default':
        h_size = 18
    fig, ax = plt.subplots(nrows=nrows, ncols=columns, figsize=(h_size,(nrows*4)+1))
    ax = ax.ravel()
    for i, column in enumerate(x_list):
        ax[i].set_title(title_list[i])
        ax[i].set_xlabel(xlab_list[i])
        ax[i].set_ylabel(ylab_list[i])
        
        type_dict = {'hist':ax[i].hist,
                 'bar':ax[i].bar,
                 'box':ax[i].boxplot,
                 'scatter':ax[i].scatter,
                 'barh':ax[i].barh}
        if type_list[i] in ['hist', 'box']:
            type_dict[type_list[i]](x_list[i])
            sup_y = 1
        else:
            type_dict[type_list[i]](x_list[i], y_list[i])
            if type_list[i] == 'bar':
                ax[i].set_xticks(x_list[i])
                ax[i].set_xticklabels(xlab_list[i])
                ax[i].set_xlabel('')
                if bar_numbers:
                    for z, v, in enumerate(y_list[i]):
                        ax[i].text(z-(len(str(v))*(.6/15)), v+0.15, str(v), color='blue', fontweight='bold')

            if type_list[i] == 'barh':
                ax[i].set_yticks(x_list[i])
                ax[i].set_yticklabels(ylab_list[i])
                ax[i].set_ylabel('')
                if bar_numbers:
                    for z, v in enumerate(y_list[i]):
                        ax[i].text(v + 0.02, z-(0.015*len(ylab_list[i])), str(v), color='blue', fontweight='bold')
                        sup_y = 1



                                        
    if len(type_list)%3 > 0 and len(type_list) != 2:
        for x in range(3 - len(type_list)%3):
            fig.delaxes(ax[-(x+1)])
    plt.suptitle(fig_title, fontsize=18, y = sup_y)
    if nrows > 1:
        fig.subplots_adjust(hspace=0.4)
                                        
    plt.show()
    
def multi_histo(dists, labels, bins=10, title='', xlabel = '', ylabel = '', show_kde=False, norm=False):
    for x in range(len(dists)):
        counts = sns.distplot(dists[x], kde=show_kde, norm_hist=norm, label=labels[x])
    counts.set_title(title)
    counts.set_xlabel(xlabel)
    counts.set_ylabel(ylabel)
    counts.legend()
    
def top_words(frame, ref, num):
    
    '''Returns a dictionary of the most frequent words from a CountVectorizer dataframe
    Frame - DataFrame: Dataframe constructed from a CountVectorizer.transform
    Ref - CountVectorizer: The CountVectorizer which generated the words of interest. Should be the same as the one which constructed the dataframe in the arguments
    Num - Int: The number of top words desired to be returned'''

    top_words = {}
    for i in ref.get_feature_names():
        top_words[i] = frame[i].sum() 

    most_freq = pd.DataFrame(sorted(top_words.items(), key = lambda x: x[1], reverse = True))
    most_freq.columns = ['word','count']
    
    if num == 0:
        return top_words
    else:
        return {most_freq['word'][x]: most_freq['count'][x] for x in range(num)}