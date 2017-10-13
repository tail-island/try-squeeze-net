import matplotlib.pyplot as plot
import pickle


def plot_history(history):
    plot.clf()
    plot.plot(history['loss'])
    plot.plot(history['val_loss'])
    plot.show()


def main():
    with open('history.pickle', 'rb') as f:
        history = pickle.load(f)

    plot_history(history)


if __name__ == '__main__':
    main()
