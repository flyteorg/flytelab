.. _model:

Machine Learning Model
---------------------------

**Objective**: To help Kinzinho define his next travel destination, we seek to find other cities similar to the last travel destination he liked the most.

**Strategy Solution**: To make a good evaluation between the cities, we chose to make a vector representation of each city in Brazil, encoding general information about each city such as its history, its geography, its climate and its tourist attractions. We chose this strategy because with a vector representation of the city we were able to apply similarity calculation operations between cities, considering various information about them.

**Input data**: For our solution we use the following data from wikipedia and wikivoyage: "summary_wikipedia_en", "history_wikipedia_pt", "geografia_wikipedia_pt", "clima_wikipedia_pt", "see_wikivoyage_en", "do_wikivoyage_en", "summary_wikivoyage_en"

**Preprocessing**: To process the data and extract only important information, we apply a series of pre-processing to clean unnecessary information and homogenize the texts.

**Model**: To generate the best vector representation of each city's features, we used a pre-trained state-of-the-art model based on Transformers (BERTimbau). As a vector representation of each feature of the city, we use the output of the last layer of the BERTimbau language model. The vector representation of each city is generated from the average of the vectors of its features.

**Similarity**: To calculate the similarity between the vector representations of each city, we are using an high optimized library and calculate the Euclidean distance between an input vector query (vector of the last city visited by Kinzinho) and all vectors of the cities available in our portfolio.


To see the dataset processing codes, access the links below.

.. toctree::
    text_preprocessing
    feature_engineering