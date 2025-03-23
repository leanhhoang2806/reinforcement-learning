
'''
sudo apt update && sudo apt install nano -y
python -m pip install -U --force pip
python -m ensurepip
export PATH=$HOME/.local/bin:$PATH

pip install torch tqdm numpy matplotlib

scp -P 31707 /home/hoang/Documents/work/reinforcement-learning/exercise/train-lstm-for-xor.py ubuntu@doting-tarragon-spider.1.cricket.hyperbolic.xyz:~


python train-lstm-for-xor.py
'''



Model Debugging step 
1. Does the model learn ? => when plot loss function over training data set, should see trending downward direction and platuea out
2. Does the model overfit or under fit ? => plot the loss function between validation set and test set
3. If the loss value is huge => mean that exploding gradients => need regulation


4. The current set up for trains-lstm-mnist.py does work. The problem is that the LSTM model does not learn anynthing on classification. It did well on regression task tho
5. The larger the model the less the epoches needed
6. the training look effective when the loss function goes down as more epoches added, and the test validation loss should pick up from where the loss function of the train set ended. The reason is that the model already made all the correction on train set, there's no fluctuation for test set.
7. the gradients over epoches also supposed to go down as well because the more epoches trained, the more the model could converge. Because the model already converge, less gradient needed, gradient should go down and close to zero when model convert
8. The gradient changed expected to be a bit larger than 0.1
9. if gradients norms approach 0. It's a strong case for vanishing gradient
10. The model seems to have multiple local minima, 
11. Sometimes, it's better to have more epoches than large model. Large model hard to transport, take time to train.
Having more epoches some what solve this problem
12. For LSTM model, the length of the input affect the depth of the model because there's more to memorized. 

13. The issue of model underfitting and approaches:
    1. 


### Odd or even LSTM case
Notes
sequence of Length 10 has better prediction

## Work Task

The current problem: network does not learn => mean that gradient has not been passed to the layers of the network
=> solution: check gradient decent values, if it's decreasing by epoches means that gradient has been vanished, Nothing to learn


How to improve LSTM when the sequence getting longer ?
what's happened to deep LSTM model ? at 512 neurons and 5 layers, accuracy is out of quack with any length of the input sequence => research on the problem and limitation of deep neural network

