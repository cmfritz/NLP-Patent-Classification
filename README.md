# Using Natural Language Processing to Classify Patents Technologies

## Business Understanding
Patents provide a wealth of information since their data has been compiled by a variety of parties in a variety of ways, both public and private. To name a few, public patent data can be accessed via websites provided by the U.S. Patent & Trademark Office (USPTO), foreign patent offices, Google Patents, etc. Data for this project has been obtained from [PatentsView.org](PatentsView.org), which is a visualization, data dissemination, and analysis platform provided by the USPTO.  

Various types of information on patents are connected at various stages in a patent's life, one being the classification of a patent's technology (e.g. mechanical, chemistry, electrical, etc.) This classification is assigned some time after a patent is filed, but it could be useful to try and predict how a patent will be classified before it is filed. In this project, I will be attempting to use machine learning to predict classifications. There are many types of classification systems ranging from broad to very specific, so for the purposes of this project, I will focus on a broader classification system.

In general, the patent-related insights and information I mention in this project come from my 10+ years of experience as a patent professional, however I also have included a list of references at the end of the readme.

## Data Understanding
To provide some background on how a patent is structured, there are 4 main parts. They are:
1. Coverpage, which provides bibliographic information such as who the inventors are, what the title is, a summary, what patents are related, etc.;
2. Specification, which provides background information on the given invention;
3. Drawings, which serve to illustrate the invention; and
4. Claims, which detail the specific aspects of an invention that the applicant wants legally protected.

Here is a link to patent [#10 million](https://patentimages.storage.googleapis.com/c0/d5/f7/86ad5b42759506/US10000000.pdf) if you are curious what an actual patent document looks like.

I mention these because in this project, we will use the first claim of U.S. patents to predict the classification. Claims are much shorter and more specific than the entire specification (a few lines vs. many pages) and follow a similar grammatical structure. This structure is a product of patent practitioners seeking to obtain specific and predictable legal protections, and may help in pattern recognition. Alternatively, the styles of the specifications are highly personalized to the author of the patent since the rules for the contents are not as rigid. Lastly, since each patent is required to contain at least one claim, we are guaranteed that there will not be missing data for any given patent we are analyzing.

## Data Preparation
### Data Selection
Currently, there have been over 10 million patents granted in the U.S., so the first thing to do is reduce the amount of data being used for modeling since I am limited to what Google Collab can run in a couple hours. The parameters I used to select the data for training are:
1. Patents from the last 10 years. This provides more than 3 million patents for further trimming. PatentsView provides a page where you can access the claims for each patent issue bye year [here](https://patentsview.org/download/claims). Each file is very large, so I limited to the past 10 years to capture current technology trends while at the same time reducing the dataset.
2. Using the World Intellectual Property Organization's (WIPO) technology fields classifications, found [here](https://patentsview.org/download/data-download-tables). Other patent classification systems are complex and hierarchical, containing hundreds of classifications. For the scale of this project, WIPO's technology fields classifications are the most approachable, having the following five categories:
    - Electrical engineering
    - Mechanical engineering
    - Instruments
    - Chemistry
    - Other
3. Getting rid of data having more than one classification. Patents that belong to more than one technology field might confuse the model.
4. Keeping only the first claim for each patent, which is usually the most important claim of a patent. While I could use all the claims, using a smaller amount of words/data should reduce the computing time required.

### Data Cleaning
Once I selected my data, I had to clean and prepare my data for modeling. My process:
1. Merged the data obtained in steps 1 and 2 above into a single dataframe.
2. Remove claims after the 1st and remove resulting duplicate entries
3. Remove classifications of "other" since they are a catch-all rather than a technology field label
4. Limited the data to 200k of each category (which was chosen by trial and error to limit computing time).
5. Prepared two versions of testing and training data:
  - 4-class version - this is as the data stands after previous steps
  - 2-class version - created by dropping two of the categories and keeping "Instruments" and "Mechanical engineering".
6. Lowercase all the words so that letter case does is not taken into account
7. Stem, remove stop words, and vectorize training and test data
 - To create the list of stop words, I relied on a standard set for the English language and also added a few words that are overly common for claim language. For example, the word "method" would be used in all disciplines. The custom words come from my personal experience working with patents.
 - I chose to stem rather than lemmatize words since lemmatizing was too compuationally demanding for this dataset.
 - I chose Tfidf for the vectorizer, since it tends to work best for NLP. I filtered out the top and bottom 10% common words. From trial and error, this percentage gave a good balance of computing time with my resources but without sacrificing too much accuracy. As I progressed in this project, computing times hours and hours became a matter of issue, so it has driven several of the decisions made in this project. It is also how I selected the length of n grams.

## Modeling & Evaluation
I wanted to look at how a few different sets of models and also compare how they would perform in dual or multiclass situations.

### Models with 2-classes

I first wanted to see how traditional machine learning models would handle 2 classes. For the dual class scenario, I kept the sector "EE" (Electrical Engineering) and pooled the other categories into "non-EE". I chose to keep EE since it was the largest class in the original dataset, which makes sense since things like electronics and computing are the biggest source of innovation currently. This 2-class prediction could be useful in a scenario, for example, where EE applications needed to be prioritized over others.

Below is visualization showing the distribution of the classes before and after cleaning.

Class distribution before:
![pic1a](./images/EE.png)

Class distribution after:
![pic1b](./images/EE_after.png)

I used the above dataframes to train 3 different models:
1. Logistic Regression
2. Decision Tree
3. XG Boost

I then boosted the best-performing model, which was XG Boost. Below is the confusion matrix and accuracy results.

Training Accuracy: 83.7%<br />
Test Accuracy: 83.2%<br />

Confusion Matrix for train data:<br />
![pic1](./images/cm_gridsearch_xg_boost_Train_2class.png)

Confusion Matrix for test data:<br />
![pic2](./images/cm_gridsearch_xg_boost_Test_2class.png)
Images by author.

From the confusion matrices, we can see that we can reliably separate the EE patents from the rest. Next, I want to look at the most influential terms. Below shows the permutation importances, which shows the top 5 most important words are:

- data
- comput (stem of computer, computing, computation, etc.)
- inform
- signal
- first (stem of device, devices, etc.)

![pic3](./images/perm_importance_xgb_4class.png)
Image by author.

### Models with 4-classes
Now that we have a sense how modeling works with 2 classes,  Below is visualization showing the distribution of the classes before and after cleaning.

Class distribution before:
![pic4a](./images/countplot.png)

Class distribution after:
![pic4b](./images/countplot_after.png)

Next I wanted to see how the same models from the dual class models would perform using all the classes. I  boosted the best-performing model, which was also XG Boost. Below is the confusion matrix and accuracy results.

Training Accuracy: 72.1%<br />
Test Accuracy: 72.0%<br />

Confusion Matrix for train data:<br />
![pic4](./images/cm_xgb_gridsearch_Train_4class.png)

Confusion Matrix for test data:<br />
![pic5](./images/cm_xgb_gridsearch_Test_4class.png)
Images by author.

The confusion matrix provides some interesting insights. While the train accuracy of 72.1% and test accuracy of 72.0% are an improvement over the previous models, we can see what may be pulling that score down. This model performs best on identifying chemistry (87.1% accuracy) and electrical (78.9% accuracy) patents but struggles more with mechanical (62.4% accuracy) and instrument patents (58.3% accuracy). (I calculated the individual accuracies manually from the test confusion matrix.) I suspect that perhaps instrument topics share vocabulary with the mechanical patents, which I will investigate by looking at the most influential features.

Below shows the permutation importances, which shows the top important words:

![pic6](./images/perm_importance_xgb_4class.png)
Image by author.

From these importances, we can see the top 5 words are:
1. data
2. comput (stem of computer, computing, computation, etc.)
3. signal
4. method
5. devic (stem of device, devices, etc.)

This is in line with my initial suspicion regarding the poor performance of the "instruments" class. These words are all typical of instrumentation but could also apply to the other disciplines. It would suggest that context is important for determining classification and would be a good candidate for a deep learning model.

## Deployment - Final Model

My final model was trained using the entire dataset and pickled in notebooks mentioned before. I decided to use the multi-class model since it was performing relatively well.

![pic7](./images/cm_Final_Model_4class.png)
Image by author.

The accuracy of the final model with the holdout data turned out to dip to 60.2%. To me, this indicates that since the holdout patent data predates the patents used in training the final model, there could be shifts in the terms or popular technologies being addressed in these patents.

## Conclusion

### Results
Based on the performance metrics for both the 2-class and 4-class models, the XG Boost model performed the best for both. As for what determined a classification, it appears to be a mix of keywords but also stylistic choices that practioners of each technology area lean towards.

### Recommendations
My goal for this project was to provide a proof of concept rather a model ready for distribution, since such a model would need to able to handle the hundreds of classifications. Based on the previous results, I feel proof of concept is shown that machine learning can be used to help automate the classification process for patents and could be useful to patent Offices or third parties, if they are not utilizing it in some fashion already.

### Future Work
Deep learning is often a good fit for natural language processing, so for my next step, I would like to finalize a deep learning model.
My preliminary but unpublished deep learning model shows extremely promising results and suggests that it could be a viable model for more complex classification systems.

For More Information, please review my full analysis in the master Jupyter notebook or my presentation.

For any additional questions, please contact Catherine Fritz: cmfritz0@gmail.com.
