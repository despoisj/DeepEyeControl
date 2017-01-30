


# DeepEyeControl
Using your eyes to trigger shortcuts on your computer.

Read the [article on Medium](https://medium.com/@juliendespois/talk-to-you-computer-with-you-eyes-and-deep-learning-a-i-odyssey-part-2-7d3405ab8be1#.9lke56u8t)

Required install:

```
tensorflow
tflearn
```

- Create folder Data/Raw/, Data/Processed/ Models/ and Datasets/

To train the classifier:

```
python model.py train
```

To test the classifier (requires some trivial tweaking in the code):

```
python model.py test
```

To use the classifier to predict current motion:

```
python main.py
```

- Most editable parameters are in the config.py file, the model can be changed in the model.py file.
- I haven't implemented the end of pipeline (actual commands trigger) but that is straightforward in the classifier.py file.
