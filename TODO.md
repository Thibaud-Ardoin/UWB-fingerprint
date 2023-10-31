Remembering the important tasks to go, the ideas and the current focus

# TODO list

- Model Save and load 

- Have a way to force the saving of the current train (?)
    - Try catching in order save the model and test it after a Ctrl+C

- Propose new architectures:
    - Adversarial position learning
    - Reset CNN model
        - Use MaxPool dim reduction
    - 16x16 signals is worth a 1000 words
    - Sliding window

- Change used optimizer:
    - Check Adamdffffff
    - Check SGD the OG of optimizers
    - Warmup ?

- Training process:
    - Try triplet loss again (eventually with covariance regularisation)
    - Could compile covariance on each position or each device individually

- Normalise loss according to batch sizes and embedding size
    - Make sure that this really works (looks like or now it doesnt..)

- Encapsulate the Losses properly in the training loop to clean all this ififififielse mess

- Separate better the train and test and validation 
    - just split also test 1st
    - Create a subset for validation (same and different training positions)
 
- Log/ plot architecture of the model for each run

- Embedding size in train is not right (?)

- Keep the best performing random seeds

- Terminate runs if they are too bad:
    - Make early stopping conditions
    - could be an early test if the loss is not mooving a lot and then decide according to the test result ?




# High level targets

- Dataset visualisation / Understanding tool (?)

- Find other nice metrics to use for evaluating model
    - Create validation process with unseen device
    - NN from on unseen position to an other ( needs multi pose evalutaiton alreaddy)


# Experimental Targets

- Propose new architectures:
    - Adversarial position (un)learning
    - Reset CNN model properly
    - 16x16 signals is worth a 1000 words - Different Transformer inputs
    - Sliding window

- Check different test position to see the most remakable

- Change used optimizer:
    - Adagrad
    - SGD
    - Check Adamdffffff
    - Warmup properly + Plateau togather ( or smooth plateau )

- Training process:
    - Could compile covariance on each position or each device individually

- Normalise loss according to batch sizes and embedding size
    - Make sure that this really works (looks like or now it doesnt..)

- Cross validate on positions while training

- Weight decay

- get compiled parameters such as (total number of trainable parameters, total lambda value) In order to see more global trends in trains

# Understandings

- Understand why a smaller extender seems to work better ?
- Is the covariance part of the loss incorrect ?
