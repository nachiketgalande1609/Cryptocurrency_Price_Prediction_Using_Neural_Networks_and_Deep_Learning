import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

# MAIN PAGE TEXT
st.write("# CRYPTOCURRENCY AND ITS FUTURE")
image = Image.open('./static/cryptocurrency_image.jpg')
st.image(image, caption='Cryptocurrency')
st.write('Welcome to the Cryptocurrency price prediction app, here you will get information about cryptocurrency like bitcoin, ethereum and litheum.')
st.title('Things to know about Cryptocurrency before investing in it:')
st.write('Trading in cryptocurrency and the stock market is a little similar. Like there, you are trading in stocks of different companies  here also, you have digital assets such as bitcoin, ethereum etc., where your trade-in. However, you do need a Demat account for the stock market, but the same is not the case with crypto exchange platforms. All you need to do is download a cryptocurrency exchange platform and then start with it.')
st.write('You have to follow the know-your-customer (KYC) process while opening an account. Barring minor differences, most exchanges require your Permanent Account Number (PAN) card and an identity proof such as your passport, Aadhaar or driving license. Some exchanges give approvals instantly. Others take up to a week to complete the KYC process.')
st.title('From which exchange should I buy a cryptocurrency?')
st.write('Their are many crypto exchange apps such as wazirx, upstox etc.')
st.write('Cryptocurrencies are unregulated instruments. Therefore, anyone can start a crypto exchange. That is why, first, you should check the background of the core team and founders of the crypto exchange, says Darshan Bathija, CEO of Vauld, a global crypto exchange and lending platform.')
st.write('New crypto exchanges will have fewer investors. They will have low liquidity with very few trades,says Nischal Shetty, Founder & CEO of WazirX, a cryptocurrency exchange')
image = Image.open('./static/wazirx.png')
st.image(image, caption = 'https://wazirx.com/invite/hekbdek7')
st.title("How do I transfer amounts to exchanges for investing?")
st.write('To buy a crypto coin, you first need to transfer money to a wallet that belongs to your exchange. Then, you can buy a coin. You can use internet banking facility (IMPS, RTGS, or NEFT) or debit cards for transferring sums. Your wallet gets credited once you submit the transaction reference number.')
st.title('What are the trading or investment charges?')
st.write('There are fees specified for trading in cryptos. Vauld charges a fee of 0.05 percent of the transaction value, whereas CoinDCX and WazirX levy a rate of 0.10 percent. Bitbns charges 0.25 percent for each transaction.')
st.title('Do I need to do any research before investing in cryptocurrency?')
st.write('Yes, of course. Apart from understanding the risk levels, you also need to decide which coin you wish to buy and why.')

# SIDEBAR
st.sidebar.header("Select Cyptocurrency")

image = Image.open('./static/bitcoin.png')
st.sidebar.image(image)
result = st.sidebar.button("BITCOIN")

image = Image.open('./static/ethereum.png')
st.sidebar.image(image)
result2 = st.sidebar.button("Ethereum")
# SIDEBAR END

if result:
    st.title("What Is Bitcoin?")
    image = Image.open('./static/bitcoin.png')
    
    # Align BITCOIN image to the center
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image(image, width=200)

    st.write("Bitcoin is a digital currency created in January 2009. It follows the ideas set out in a whitepaper by the mysterious and pseudonymous Satoshi Nakamoto.The identity of the person or persons who created the technology is still a mystery. Bitcoin offers the promise of lower transaction fees than traditional online payment mechanisms and, unlike government-issued currencies, it is operated by a decentralized authority.")

    data = pd.read_csv('BTC-USD.csv', date_parser=True)

    # STREAMLIT HEADLINES
    st.title("Bitcoin Price Prediction")
    st.write("Data Collected from https://in.investing.com/crypto/bitcoin/historical-data")
    st.table(data.head())

    data_training = data[data['Date'] < '2020-10-01'].copy()

    data_test = data[data['Date'] > '2021-02-16'].copy()
    training_data = data_training.drop(['Date', 'Adj Close'], axis=1)

    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)

    X_train = []
    Y_train = []

    # training_data.shape[0]

    for i in range(60, training_data.shape[0]):
        X_train.append(training_data[i - 60:i])
        Y_train.append(training_data[i, 0])
    
    X_train, Y_train = np.array(X_train), np.array(Y_train)

    from tensorflow import keras
    from keras.layers import Dense, LSTM, Dropout
    from keras import Sequential

    regressor = Sequential()
    regressor.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 5)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=60, activation='relu', return_sequences=True))
    regressor.add(Dropout(0.3))

    regressor.add(LSTM(units=80, activation='relu', return_sequences=True))
    regressor.add(Dropout(0.4))

    regressor.add(LSTM(units=120, activation='relu'))
    regressor.add(Dropout(0.5))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("Training the model..."):
        regressor.fit(X_train, Y_train, epochs=1, batch_size=50)
    st.success("Training complete!")

    past_60_days = data_training.tail(60)
    df = pd.concat([past_60_days, data_test], ignore_index=True)
    df = df.drop(['Date', 'Adj Close'], axis=1)
    df
    inputs = scaler.transform(df)
    X_test = []
    Y_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60:i])
        Y_test.append(inputs[i, 0])

    X_test, Y_test = np.array(X_test), np.array(Y_test)
    X_test.shape, Y_test.shape

    X_test.shape
    Y_pred = regressor.predict(X_test)
    Y_pred = Y_pred.reshape(-1)  # Reshape to (150,)
    Y_pred.shape

    # Y_pred, Y_test

    scale = 1 / scaler.scale_[0]
    Y_test = Y_test * scale
    Y_pred = Y_pred * scale

    # Y_pred

    # Y_test

    st.title('Prediction')
    st.write('Bitcoin Predicted price can be seen from graph below. Graph has been created using LSTM algorithm.')
    st.write('Below predicted price is predicted using last 365 days of bitcoin prices')
    st.write('Graph show the predicted price from today till next 150 days')
    plt.figure(figsize=(14, 5))
    plt.plot(Y_test, color='red', label='Real Bitcoin Price')
    plt.plot(Y_pred, color='green', label='Predicted Bitcoin Price')
    plt.title('Bitcoin Pridicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    # st.pyplot(plt.gcf())
    st.pyplot(plt)

    st.write('Note: Above is the predicted price which is predicted by our system')
    st.write('Copyright: MIT ADT University')
    

if result2:
    st.title("What is Ethereum")
    image = Image.open('./static/ethereum.png')
    
    # Align ETHEREUM image to the center
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image(image, width=200)
    
    st.write('Ethereum is a technology thats home to digital money, global payments, and applications. The community has built a booming digital economy, bold new ways for creators to earn online, and so much more. Its open to everyone, wherever you are in the world – all you need is the internet.')

    data = pd.read_csv('C:/Users/nachi/OneDrive/Desktop/FinalYearProject/Code/ETH-USD.csv', date_parser=True)
    
    st.title("Ethereum Price Prediction")
    st.write("Data Collected from https://in.investing.com/crypto/ethereum/historical-data")
    st.table(data.head())
    
    data_training = data[data['Date'] < '2020-10-01'].copy()

    data_test = data[data['Date'] > '2021-02-01'].copy()
    training_data = data_training.drop(['Date', 'Adj Close'], axis=1)

    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)

    X_train = []
    Y_train = []

    training_data.shape[0]

    for i in range(60, training_data.shape[0]):
        X_train.append(training_data[i - 60:i])
        Y_train.append(training_data[i, 0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    X_train.shape

    from tensorflow import keras
    from keras.layers import Dense, LSTM, Dropout
    from keras import Sequential

    regressor = Sequential()
    regressor.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 5)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=60, activation='relu', return_sequences=True))
    regressor.add(Dropout(0.3))

    regressor.add(LSTM(units=80, activation='relu', return_sequences=True))
    regressor.add(Dropout(0.4))

    regressor.add(LSTM(units=120, activation='relu'))
    regressor.add(Dropout(0.5))

    regressor.add(Dense(units=1))

    regressor.summary()

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("Training the model..."):
        regressor.fit(X_train, Y_train, epochs=1, batch_size=50)
    st.success("Training complete!")

    past_60_days = data_training.tail(60)
    df = pd.concat([past_60_days, data_test], ignore_index=True)
    df = df.drop(['Date', 'Adj Close'], axis=1)
    st.table(df.head())

    inputs = scaler.transform(df)

    X_test = []
    Y_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60:i])
        Y_test.append(inputs[i, 0])

    X_test, Y_test = np.array(X_test), np.array(Y_test)
    X_test.shape, Y_test.shape

    Y_pred = regressor.predict(X_test)
    # Y_pred, Y_test

    scaler.scale_

    scale = 1 / 5.18164146e-05
    scale

    Y_test = Y_test * scale
    Y_pred = Y_pred * scale

    Y_pred
    
    Y_test

    st.title('Prediction')
    st.write('Ethereum Predicted price can be seen from graph below. Graph has been created using LSTM algorithm.')
    st.write('Below predicted price is predicted using last 365 days of bitcoin prices')
    st.write('Graph show the predicted price from today till next 150 days')

    plt.figure(figsize=(14, 5))
    plt.plot(Y_test, color='red', label='Real Ethereum Price')
    plt.plot(Y_pred, color='green', label='Predicted Ethereum Price')
    plt.title('Ethereum Price Prediction using RNN-LSTM')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    # st.pyplot(plt.gcf())
    st.pyplot(plt)

    st.write(' ')
    st.write('Note: Above is the predicted price which is predicted by our system')
    st.write('Copyright: MIT ADT University')