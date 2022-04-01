# whats_cooking_good_looking

## Problem statement
The problem is inspired from a past project where we were supposed to extract brands from social media posts.


## Target Solution implementation
In order to extract brands in a tweet, we implemented a NER (named entity recognition) pipeline that:
1. Retrieves tweets related to beauty based on keywords
2. Applies a pretrained NER model to those posts
3. Sends model results to a labelling interface and waits for a manual annotation to check given results
4. Computes evaluation metrics (so far only accuracy but it would be interesting to compute precision and recall as well)
5. 1. If metrics are good enough (defined by a business standard), the pipeline end
5. 2. If metrics are not good enough, it sends those labeled posts into a training task and goes back into the same piece of pipeline composed of steps 2 - 3 - 4

<p align="center">
    <img src="/docs/functional_pipeline.png" />
</p>

## Actual Solution implementation

The project is cut into 3 steps:

### 1. NER application pipeline
1. Retrieves tweets related to beauty based on keywords
2. Applies a NER model to those posts
3. Sends model results in a format that could load into Label Studio in a GCS bucket

To run this pipeline locally please run
```python whats_cooking_good_looking/apply_ner_workflow.py```

### 2. Manual labelling part in Label Studio

<<<< Insert Label Studio documentation and images >>>>

### 3. NER training pipeline
1. Retrieves labelled tasks (Label Studio output)
2. Computes model accuracy based on those labelled observations
3. 1. If metrics are good enough, pipeline stops
3. 2. If metrics are not good enough, labelled tasks are used as input to train a new NER model

The goal was to create a feedback loop where it was possible to iterate by training NER models based on new manual annotations. We chose to cut into 2 pipelines to get rid of the network constraints that we would have to handle and that were not evaluated in the scope of this hackaton.

To run this pipeline locally please run
```python whats_cooking_good_looking/apply_ner_workflow.py```

## Pipeline deployment

<<<< Insert github action explanation >>>>
