import random
from collections import deque
from enum import Enum
from typing import List
import numpy as np
import stock_exchange
from experts.obscure_expert import ObscureExpert
from framework.vote import Vote
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.order import Order, OrderType
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from framework.order import Company
from framework.utils import save_keras_sequential, load_keras_sequential
from framework.logger import logger


class DeepQLearningTrader(ITrader):
    """
    Implementation of ITrader based on Deep Q-Learning (DQL).
    """
    RELATIVE_DATA_DIRECTORY = 'traders/dql_trader_data'

    def __init__(self, expert_a: IExpert, expert_b: IExpert, load_trained_model: bool = True,
                 train_while_trading: bool = False, color: str = 'black', name: str = 'dql_trader', ):
        """
        Constructor
        Args:
            expert_a: Expert for stock A
            expert_b: Expert for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
        """

        # Save experts, training mode and name
        super().__init__(color, name)
        assert expert_a is not None and expert_b is not None
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.train_while_trading = train_while_trading

        #self.actions = [
            #(+0.0, +1.0),
            #(+0.0, -1.0),
            #(+0.0, +0.0),
            #(+1.0, +0.0),
            #(-1.0, +0.0),
            #(-1.0, -1.0),
            #(-1.0, +1.0),
            #(+1.0, -1.0),
            #(+0.25, +0.75),
            #(+0.50, +0.50),
            #(+0.75, +0.25),
            #(-0.25, -0.75),
            #(-0.50, -0.50),
            #(-0.75, -0.25),
            #(-0.50, +0.50),
            #(+0.50, -0.50),
            #(-0.25, +0.25),
            #(+0.25, -0.25)
        #]

        # Parameters for neural network
        self.state_size = 2
        self.action_size = 9
        self.hidden_size = 50

        # Parameters for deep Q-learning
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 8  # 1#64
        self.min_size_of_memory_before_training = 125  # 5#1000  # should be way bigger than batch_size, but smaller than memory
        self.memory = deque(maxlen=250)  # 10)#2000)
        self.gamma = 0.9

        # Attributes necessary to remember our last actions and fill our memory with experiences
        self.last_state = None
        self.last_action = None
        self.last_portfolio_value = None

        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        self.model = None
        if load_trained_model:
            self.model = load_keras_sequential(self.RELATIVE_DATA_DIRECTORY, self.get_name())
            logger.info(f"DQL Trader: Loaded trained model")
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='relu'))
            self.model.add(Dense(self.hidden_size, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this traders.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.get_name())
        logger.info(f"DQL Trader: Saved trained model")

    def current_state(self, stock_market_data: StockMarketData):
        stock_data_a = stock_market_data[Company.A]
        stock_data_b = stock_market_data[Company.B]
        vote_a = self.expert_a.vote(stock_data_a)

        vote_b = self.expert_b.vote(stock_data_b)

        return np.array([[self.vote_mapping(vote_a.value),
                          self.vote_mapping(vote_b.value)]])

    def reward(self, current_portfolio_value: float):
        if self.last_portfolio_value < current_portfolio_value:
            return 10 * (current_portfolio_value / self.last_portfolio_value)
        elif self.last_portfolio_value > current_portfolio_value:
            return -10

        return 0

    def experience_replay(self):
        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state in batch:
            target_qval = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target = self.model.predict(state)
            target[0][action] = target_qval
            self.model.fit(state, target, epochs=1, verbose=0)

    def vote_mapping(self, vote):
        if vote == "sell":
            return -1
        elif vote == "hold":
            return 0
        else:
            return 1


    def get_action(self, state):
        if (np.random.rand() <= self.epsilon) and (self.epsilon >= self.epsilon_min) and self.train_while_trading:
            action = np.random.randint(self.action_size)
        else:
            prediction = self.model.predict(state)
            action = np.argmax(prediction[0])
        return action

    def mapping_action_order(self, action, portfolio, stock_market_data: StockMarketData):
        all_orders = []

        sell_price_A = portfolio.get_stock(Company.A)
        sell_price_B = portfolio.get_stock(Company.B)
        
        stock_price_A = stock_market_data[Company.A].get_last()[-1]
        stock_price_B = stock_market_data[Company.B].get_last()[-1]
        
        buy_price_A = int(portfolio.cash // stock_price_A)
        buy_price_B = int(portfolio.cash // stock_price_B)

        if action == 0:
            invest = int(portfolio.cash // 2)
            buy_price_A = int(invest // stock_price_A)
            all_orders.append(Order(OrderType.BUY, Company.A, buy_price_A))
            buy_price_B = int(invest // stock_price_B)
            all_orders.append(Order(OrderType.BUY, Company.B, buy_price_B))
        elif action == 1:
            all_orders.append(Order(OrderType.BUY, Company.A, buy_price_A))
            all_orders.append(Order(OrderType.SELL, Company.B, sell_price_B))
        elif action == 2:
            all_orders.append(Order(OrderType.BUY, Company.A, buy_price_A))
        elif action == 3:
            all_orders.append(Order(OrderType.SELL, Company.A, sell_price_A))
            all_orders.append(Order(OrderType.BUY, Company.B, buy_price_B))
        elif action == 4:
            all_orders.append(Order(OrderType.SELL, Company.A, sell_price_A))
            all_orders.append(Order(OrderType.SELL, Company.B, sell_price_B))
        elif action == 5:
            all_orders.append(Order(OrderType.SELL, Company.A, sell_price_A))
        elif action == 6:
            all_orders.append(Order(OrderType.BUY, Company.B, buy_price_B))
        elif action == 7:
            all_orders.append(Order(OrderType.SELL, Company.B, sell_price_B))
        elif action == 8:
            all_orders = all_orders

        return all_orders

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate action to be taken on the "stock market"

        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation
        Returns:
          A OrderList instance, may be empty never None
        """
        assert portfolio is not None
        assert stock_market_data is not None
        assert stock_market_data.get_companies() == [Company.A, Company.B]

        # Compute the current state
        current_state = self.current_state(stock_market_data)
        current_portfolio_value = portfolio.get_value(stock_market_data)

        # Store state as experience (memory) and train the neural network only if trade() was called before at least once
        if self.last_action is not None and self.train_while_trading:
            reward = self.reward(current_portfolio_value)
            self.memory.append((self.last_state, self.last_action, reward, current_state))

            if len(self.memory) > self.min_size_of_memory_before_training:
                self.experience_replay()

        # Create actions for current state and decrease epsilon for fewer random actions
        action = self.get_action(current_state)
        self.epsilon = max([self.epsilon_min, self.epsilon * self.epsilon_decay])

        # Save created state, actions and portfolio value for the next call of trade()
        self.last_state = current_state
        self.last_action = action
        self.last_portfolio_value = current_portfolio_value

        # convert action to orderlist
        return self.mapping_action_order(action, portfolio, stock_market_data)


# This method retrains the traders from scratch using training data from TRAINING and test data from TESTING
EPISODES = 5
if __name__ == "__main__":
    # Create the training data and testing data
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    training_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    testing_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    # Create the stock exchange and one traders to train the net
    stock_exchange = stock_exchange.StockExchange(10000.0)
    training_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), False, True)

    # Save the final portfolio values per episode
    final_values_training, final_values_test = [], []

    for i in range(EPISODES):
        logger.info(f"DQL Trader: Starting training episode {i}")

        # train the net
        stock_exchange.run(training_data, [training_trader])
        training_trader.save_trained_model()
        final_values_training.append(stock_exchange.get_final_portfolio_value(training_trader))

        # test the trained net
        testing_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False)
        stock_exchange.run(testing_data, [testing_trader])
        final_values_test.append(stock_exchange.get_final_portfolio_value(testing_trader))

        logger.info(f"DQL Trader: Finished training episode {i}, "
                    f"final portfolio value training {final_values_training[-1]} vs. "
                    f"final portfolio value test {final_values_test[-1]}")

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(final_values_training, label='training', color="black")
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['training', 'test'])
    plt.show()
