import matplotlib.pyplot as plot
import pickle

def plot_history(history):
    plot.clf()
    plot.plot(history['acc'])
    plot.plot(history['val_acc'])
    plot.show()


def main():
    with open('history.pickle', 'rb') as f:
        history = pickle.load(f)

    plot_history(history)


if __name__ == '__main__':
    main()
