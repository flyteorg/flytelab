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
