class LSTMBettingAgent(BettingAgent):
    def __init__(self, id, name, lengthOfRace, endOfInPlayBettingPeriod, influenced_by_opinions,
                 local_opinion, uncertainty, lower_op_bound, upper_op_bound):
        super().__init__(id, name, lengthOfRace, endOfInPlayBettingPeriod, influenced_by_opinions,
                         local_opinion, uncertainty, lower_op_bound, upper_op_bound)
        self.model = load_model('/home/ubuntu/trained_lstm_model.h5')
        logging.info("LSTM model loaded successfully.")
        
        self.scaler = joblib.load('/home/ubuntu/scaler.joblib')
        logging.info("Scaler loaded successfully.")

        self.bettingInterval = 2
        self.bettingTime = random.randint(5, 15)
        self.name = 'LSTMBettingAgent'

    def getorder(self, time, markets):
        order = None
        if len(self.orders) > 0:
            order = self.orders.pop()
        return order

    def make_decision(self, time, stake, distance, rank):
        try:
            # Create a DataFrame with the input features
            data = pd.DataFrame({
                'competitorID': [self.id],
                'time': [time],
                'exchange': [0],  # Assuming exchange 0 for simplicity
                'odds': [0],  # Placeholder as odds need to be fetched from markets
                'agentID': [self.id],
                'stake': [stake],
                'distance': [distance],
                'rank': [rank],
                'balance': [self.balance],
                'label': [0]  # Include the label feature with a dummy value
            })

            # Scale the input data
            scaled_data = self.scaler.transform(data)

            # Reshape for LSTM model input
            X_scaled = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))

            # Make a prediction using the LSTM model
            prediction = self.model.predict(X_scaled)
            decision = 1 if prediction > 0.5 else 0

            logging.info(f"Decision: {decision}, Prediction: {prediction[0][0]}")
            return decision
        except Exception as e:
            logging.error(f"Error making decision: {e}")
            raise e

    def respond(self, time, markets, trade):
        if self.bettingPeriod == False: return None
        order = None
        if self.raceStarted == False: return order

        if self.bettingTime <= self.raceTimestep and self.raceTimestep % self.bettingInterval == 0:
            sortedComps = sorted((self.currentRaceState.items()), key=operator.itemgetter(1))
            
            for rank, (competitor, distance) in enumerate(sortedComps):
                decision = self.make_decision(time, 15, distance, rank+1)
                if decision == 1:  # Decision = back
                    if markets[self.exchange][competitor]['backs']['n'] > 0:
                        quoteodds = max(MIN_ODDS, markets[self.exchange][competitor]['backs']['best'] - 0.1)
                    else:
                        quoteodds = markets[self.exchange][competitor]['backs']['worst']

                    order = Order(self.exchange, self.id, competitor, 'Back', quoteodds,
                                random.randint(self.stakeLower, self.stakeHigher),
                                markets[self.exchange][competitor]['QID'], time)

                    if order.direction == 'Back':
                        liability = self.amountFromOrders + order.stake
                        if liability > self.balance:
                            continue
                        else:
                            self.orders.append(order)
                            self.amountFromOrders = liability

                elif decision == 0:  # Decision = lay
                    if markets[self.exchange][competitor]['lays']['n'] > 0:
                        quoteodds = markets[self.exchange][competitor]['lays']['best'] + 0.1
                    else:
                        quoteodds = markets[self.exchange][competitor]['lays']['worst']

                    order = Order(self.exchange, self.id, competitor, 'Lay', quoteodds,
                                random.randint(self.stakeLower, self.stakeHigher),
                                markets[self.exchange][competitor]['QID'], time)

                    if order.direction == 'Lay':
                        liability = self.amountFromOrders + ((order.stake * order.odds) - order.stake)
                        if liability > self.balance:
                            continue
                        else:
                            self.orders.append(order)
                            self.amountFromOrders = liability

        return None

