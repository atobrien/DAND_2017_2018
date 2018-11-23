
# coding: utf-8

# ## Analyze A/B Test Results
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[2]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[3]:


df = pd.read_csv("ab_data.csv")
df.head()


# b. Use the below cell to find the number of rows in the dataset.

# In[4]:


df.info()
#there are 294,478 rows


# c. The number of unique users in the dataset.

# In[5]:


df.nunique()
# there are 290584 unique users. suggesting 
# that the same user has various visits to the platform


# d. The proportion of users converted.

# In[6]:


df.groupby(['converted']).nunique()


# In[7]:


35173/290584
# the proportion of users converted is approximately 12%


# e. The number of times the `new_page` and `treatment` don't line up.

# In[8]:


df.groupby(['landing_page', 'group']).count()


# In[9]:


1965+1928
# the number of times the new_page and treatment don't line up is 1928, 
# the number of times treatment and new_page don't aline is 1965
# the total is 3893


# f. Do any of the rows have missing values?

# In[10]:


df.info()


# According to the information provided in df.info() there are no missing values
# in this dataset

# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# **How should we handle the rows where the landing_page and group columns don't align?**
# 
# * We should remove these rows

# In[11]:


# I use query to select: 
# 1) the treatment & new page rows, 
# 2) the control & old page rows.
# then I use append function to join these two groups of information
# and create a new dataframe with only this information called df2

df_a= df.query('group == "treatment"').query('landing_page == "new_page"')
df_b= df.query('group == "control"').query('landing_page == "old_page"')
df2=df_a.append(df_b, ignore_index=True)


# In[12]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[13]:


df.nunique()
# there are 290,584 unique user_ids


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[14]:


df2[df2.duplicated(subset="user_id", keep=False)]
# I use the duplicated function and the subset 
# user_id, and keep=False to show the 
# one user which is duplicated in the df2 dataframe


# c. What is the row information for the repeat **user_id**? 

# * **The landing_page for the non-unique id**: new_page
# * **The group for the non-unique id**: treatment
# * **The value of converted column for the non-unique id**: 0

# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[15]:


df2.drop(1404, inplace=True)
# I use the drop function to remove the duplicate user with id 1404


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[16]:


df2.groupby("converted").count()


# In[17]:


#probability of converting is approximately 11.9%
34753/(255831+34753)


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[18]:


df2.groupby(["group", "converted"]).count()


# In[19]:


#probability of converting given that the individual was in the 
#the control group is about 12%
17489/(17489+127785)


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[20]:


#probability of converting given that the individual was in the
#the treatment group is about 11.8%
17264/(17264+128046)


# d. What is the probability that an individual received the new page?

# In[21]:


df2.groupby(["landing_page"]).count()


# In[22]:


# the probability that an individual recieved a new page is approximately 50%
145310/(145310+145274)


# e. Consider your results from a. through d. above, and explain below whether you think there is sufficient evidence to say that the new treatment page leads to more conversions.

# **Based solely on descriptive statistics, it seems that there is not a significant difference between individuals converting relative to the group they are designated to.**

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **Null hypothesis**: 
# 
# 1) The probability of converting on the new page is less than or the same as converting on the old page. 
# 
# H0: $p_{new}$ <= $p_{old}$
# 
# **Alternative hypothesis**: 
# 
# 2) The probability of converting on the new page is greater than converting on the old page.
# 
# H1: $p_{new}$ > $p_{old}$

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[23]:


# Using query we calculate the following proportion: number of total 
# unique converted divded by the number of total unique users in new group
# i.e. # total unique converted/# total unique users
p_new = df2.query('converted == 1')['user_id'].nunique()/df2['user_id'].nunique()
p_new


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[24]:


# Using query we calculate the following proportion: number of total 
# unique converted divded by the number of total unique users in old group
# i.e. # total unique converted/# total unique users
p_old = df2.query('converted == 1')['user_id'].nunique()/df2['user_id'].nunique()
p_old


# c. What is $n_{new}$?

# In[25]:


# this is the total number of individuals in the treatment group
n_new= df2.query('group == "treatment"').shape[0]
n_new


# d. What is $n_{old}$?

# In[26]:


# this is the total number of individuals in the old group
n_old= df2.query('group == "control"').shape[0]
n_old


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[100]:


# Randomly alocate 0s and 1s with probability p_new under 
# the null, in an array of a size n_new.
new_page_converted = np.random.binomial(1, p_new, n_new)


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[101]:


# Randomly alocate 0s and 1s with probability p_old under 
# the null, in an array of a size n_old
old_page_converted = np.random.binomial(1, p_old, n_old)


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[102]:


# Here I calculate the observed difference between the Pnew and Pold 
new_page_p= new_page_converted.mean() 
old_page_p= old_page_converted.mean() 

diff1= new_page_p-old_page_p
diff1


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in a numpy array called **p_diffs**.

# In[110]:


# Here I bootstrap 10,000 times a sample from the probabilities and number 
# of events calculated from a. through g.
new_converted_simulation = np.random.binomial(n_new, p_new, 10000)/n_new
old_converted_simulation = np.random.binomial(n_old, p_old, 10000)/n_old
p_diffs = new_converted_simulation - old_converted_simulation


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[111]:


plt.hist(p_diffs);


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[113]:


#Calculate the observed difference
p_new_obs = df_a.query('converted == 1')['user_id'].nunique()/df_a['user_id'].nunique()
p_old_obs = df_b.query('converted == 1')['user_id'].nunique()/df_b['user_id'].nunique()
diff_obs= p_new_obs-p_old_obs


# In[114]:


(diff_obs < p_diffs).mean()


# k. In words, explain what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **In part j. I am testing the probability of how "extreme" or more "extreme" is the value obtained as the difference between p_new and p_old, given that the null hypothesis is true. In other words by creating a sampling distribution, which is normalized under 10,000 bootstraps, I am seeing if the observed difference calculated (diff_obs= p_new_obs-p_old_obs) is far enough from the sample's mean value to say that it is statistically different from the sampling distribution. In this case the value is not extreme enough and we do not have enough data to reject the null hypothesis that the number of converted are different between the treatment and control group.**

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[100]:


import statsmodels.api as sm
old = df2[df2['group']=='control']
new = df2[df2['group']=='treatment']
convert_old = np.sum(old['converted']==1)
convert_new = np.sum(new['converted']==1)
n_old = df2.query('group == "control"').shape[0]
n_new = df2.query('group == "treatment"').shape[0]


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[113]:


from scipy.stats import norm
sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], value=None, alternative='smaller', prop_var=False)
# this defines the values for the 95% confidence interval


# In[114]:


z_score, p_value= sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], value=None, alternative='smaller', prop_var=False)
z_score


# In[115]:


norm.cdf(z_score)


# In[116]:


norm.ppf(1-(0.05/2))


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **Since the z-score of 0.90505831275902449 does not exceed the critical value of 1.959963984540054, we do not have enough data to reject the null hypothesis that the difference between the two proportions is no different from zero. Therefore we can say that there is not a statistically significant difference between the number of converted between the treatment and control groups.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Logistic regression (since the independent value is binomial)**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[115]:


df2.head()


# In[169]:


df2['intercept'] = 1
ab_page = pd.get_dummies(df2['group'])
df3 = df2.join(ab_page, lsuffix='', rsuffix='')
df3.head()


# In[170]:


# I created a separate dataframe (df3) to run in python natively
#with scipy 0.19.0 for logistic regression section
df3.to_csv('logit_abtest.csv', index=False)


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[171]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

logit = sm.Logit(df3[['converted']], df3[['intercept', 'control']])
results = logit.fit()
# results.summary I could not run this in jupyter so I ran nateively
# in python shell, and included an image of the results


# In[194]:


from IPython.display import Image
Image(filename="C:/Users/Anthony O'Brien/Desktop/Presentation1.png")


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# **The logistic model using number of converted as the independent variable, and the control/treatment group as the dependent variable is not significant. The dependent variable does not significantly predict the Independent variable.**

# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# **The null hypothesis is that the group can not predict the conversion, while the alternative hypothesis states that the group predicts the conversion**
# 
# **The p-value for the model is 0.1899. We are attempting to see if there is a significant difference between converted and the group in which the participant is in, i.e. if the group can predict the outcome. That is to say, we are looking at a two-sided test, which is only interested in studying if there is a statistically significant difference (without considering directiona)**
# 
# **While the original test performed assumed a distribution and was used to see if the percentages between the groups were different (not if they can predict the conversion). Additionally the original test is interested in a direction (i.e. "greater than"/one-sided). One can observe this difference when converting from the two-sided to the one-sided p_value: 1-(0.19/2) = 0.9**

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Inclusion of additional covariates into a model can provide a deeper understanding of how variables predict an outcome. For example the change of group's behaviour over time (i.e. interaction of group and time) can predict an outcome at different moments. That is, at point A two groups may be the same, but after an intervention at point B the groups may differ. Therefore univariate analysis alone, while extremely relevant is only limited to one-dimensional aspects, and in general real life outcomes are more complex**
# 
# **On the other hand inclusion of multiple covariates into a model makes it difficult to understand the meaning of the results (i.e. difficult to interpret), and can also lead to scenarios like over-fitting which can lead to false positive results (type 1 error)** 

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[186]:


countries_df = pd.read_csv('./countries.csv')
df_new = countries_df.set_index('user_id').join(df3.set_index('user_id'), how='inner')


# In[ ]:


### Create the necessary dummy variables


# In[188]:


country = pd.get_dummies(df_new['country'])
df_new2 = df_new.join(country, lsuffix='', rsuffix='')


# In[190]:


df_new2.head()


# In[69]:


df_new2.to_csv('logit_abtest2.csv', index=False)


# In[191]:


logit = sm.Logit(df_new2[['converted']], df_new2[['control', 'CA', 'UK', 'intercept']])
results = logit.fit()
# results.summary I could not run this in jupyter so I ran nateively
# in python shell, and included an image of the results


# In[195]:


from IPython.display import Image
Image(filename="C:/Users/Anthony O'Brien/Desktop/Presentation2.png")


# The country dummy variables are not significantly related to the conversion rate as univariate covariates in the model.

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[192]:


# I create a new column for the interaction between the group and the country
df_new2['ctr_ca'] = df_new2["control"]*df_new2["CA"]
df_new2['ctr_uk'] =df_new2["control"]*df_new2["UK"]


# In[193]:


df_new2.head()


# In[86]:


logit = sm.Logit(df_new2[['converted']], df_new2[['control', 'CA', 'UK', 'ctr_ca', 'ctr_uk','intercept']])
results = logit.fit()


# In[87]:


# results.summary I could not run this in jupyter so I ran nateively
# in python shell, and included an image of the results


# In[196]:


from IPython.display import Image
Image(filename="C:/Users/Anthony O'Brien/Desktop/Presentation3.png")


# In[121]:


### Fit Your Linear Model And Obtain the Results


# **The interactions are not significnat. However the method used to add this information to the model is not standardized (eg. step-up model) and therefore more rigourous modeling techniques are needed to understand this relationship. As an exploratory analysis the outcome in conjunction with the previous tests suggest that the conversion rate is not predicted by the group or the country covariates.**

# <a id='conclusions'></a>
# ## Conclusions
# 
# Congratulations on completing the project! 
# 
# ### Gather Submission Materials
# 
# Once you are satisfied with the status of your Notebook, you should save it in a format that will make it easy for others to read. You can use the __File -> Download as -> HTML (.html)__ menu to save your notebook as an .html file. If you are working locally and get an error about "No module name", then open a terminal and try installing the missing module using `pip install <module_name>` (don't include the "<" or ">" or any words following a period in the module name).
# 
# You will submit both your original Notebook and an HTML or PDF copy of the Notebook for review. There is no need for you to include any data files with your submission. If you made reference to other websites, books, and other resources to help you in solving tasks in the project, make sure that you document them. It is recommended that you either add a "Resources" section in a Markdown cell at the end of the Notebook report, or you can include a `readme.txt` file documenting your sources.
# 
# ### Submit the Project
# 
# When you're ready, click on the "Submit Project" button to go to the project submission page. You can submit your files as a .zip archive or you can link to a GitHub repository containing your project files. If you go with GitHub, note that your submission will be a snapshot of the linked repository at time of submission. It is recommended that you keep each project in a separate repository to avoid any potential confusion: if a reviewer gets multiple folders representing multiple projects, there might be confusion regarding what project is to be evaluated.
# 
# It can take us up to a week to grade the project, but in most cases it is much faster. You will get an email once your submission has been reviewed. If you are having any problems submitting your project or wish to check on the status of your submission, please email us at dataanalyst-project@udacity.com. In the meantime, you should feel free to continue on with your learning journey by beginning the next module in the program.

# ## References
# 
# https://stats.idre.ucla.edu/other/mult-pkg/faq/pvalue-htm/
