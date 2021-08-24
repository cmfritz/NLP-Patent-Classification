# Using NLP to predict patent classification

## Business Understanding
Patents provide a wealth of data since their data has been compiled by a variety of parties in a variety of ways, both public and private. To name a few, public patent data can be accessed via websites for the U.S. Patent & Trademark Office (USPTO), foreign patent offices, Google Patents, etc. Data for this project has been obtained from [PatentsView.org](PatentsView.org), which is a visualization, data dissemination, and analysis platform provided by the USPTO.

Various types of information on patents are connected at various stages in a patent's life, one being the classification of a patent's technology (e.g. mechanical, chemistry, electrical, etc.) This classification is assigned some time after a patent is filed, but it could be useful to try and predict how a patent will be classified before it is filed. In this project, I will be attempting to use machine learning to predict classifications. There are many types of classification systems ranging from broad to very specific, so for the purposes of this project, I will focus on a broader classification system.

## Data Understanding
To provide some background on how a patent is structured, there are 4 main parts. They are:
1) Coverpage, which provides bibliographic information such as who the inventors are, what the title is, a summary, what patents are related, etc.; 
2) Specification, which provides background information on the given invention;
3) Drawings, which serve to illustrate the invention; and
4) Claims, which detail the specific aspects of an invention that the applicant wants legally protected. 

Here is a link to patent [#10 million](https://patentimages.storage.googleapis.com/c0/d5/f7/86ad5b42759506/US10000000.pdf) if you are curious what the actual document looks like.

I mention these because in this project, we will use the first claim of U.S. patents to predict the classification. Claims are much shorter and more specific than the entire specification (a few lines vs. many page) and follow a similar grammatical structure. This structure is a product of patent attorneys seeking to obtain specific and predictable legal protections, and may help in pattern recognition. Alternatively, the styles of the specifications are highly personalized to the author of the patent since the rules for the contents are not as rigid. Lastly, since each patent is required to contain at least one claim, we are guaranteed that there will not be missing data for any given patent we are analyzing. 

## Data Preparation
Currently, there have been over 10 million patents granted in the U.S., so the first thing to do is reduce the amount of data being used for modeling since I am limited to what Google Collab can run in a couple hours. The parameters I used to select the data for training are:
1) Patents from the last 10 years. This provides more than 3 million patents for further trimming.
2) Using the World Intellectual Property Organization's (WIPO) classifications, found [here](https://patentsview.org/download/data-download-tables), which are the broadest having only 5 classifications:
    - Electrical engineering
    - Mechanical engineering
    - Instruments
    - Chemistry 
    - Other (which patents I eliminated since I wanted to focus on data with an actual label)
3) Getting rid of data having more than one classification
The first claim is typically considered the most important. A smaller amount of data should reduce the computing power required.
4) Keeping only the first claim for each patent, which is usually the most important claim of a patent.

## Modeling & Evaluation
I wanted to look at how a few different sets of models and also compare how they would perform in dual or multiclass situations. 

### Multiclass
#### Model 1 - Logistic Regression
For my first model, I used logistic regression, and below are the results from the model.

Training Accuracy: 62.67%<br />
Test Accuracy: 62.71%

Confusion Matrix for test data:<br />
![pic1](./images/cm_logistic_regression_4class.png)

#### Model 2 - Decision Tree
For my second model, I used a decision tree classifier. Below are the results.

Training Accuracy: 96.12%<br />
Test Accuracy: 58.58%

Confusion Matrix for test data:<br />
![pic2](./images/cm_decision_tree_4class.png)

#### Model 3 - XG Boost
#### Model 4 - Nearest Neighbors

### Dualclass
#### Model 1 - Logistic Regression
#### Model 2 - Decision Tree
#### Model 3 - XG Boost
#### Model 4 - Nearest Neighbors
#### Model 5 - Deep Learning

## Evaluation

## Conclusion
### Results
### Recommendations

## Deployment
For More Information, please review my full analysis in Jupyter Notebook or my presentation.

For any additional questions, please contact Catherine Fritz: cmfritz0@gmail.com.

## Repository Structure