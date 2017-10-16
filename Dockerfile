FROM eywalker/tensorflow-jupyter:v1.0.0rc0

RUN pip3 install seaborn
RUN pip3 install -U scikit-learn
RUN pip3 install pandas_datareader
RUN pip3 install imblearn
RUN pip3 install xgboost
RUN pip3 install scipy

WORKDIR /src

ADD . /src/local_lr
RUN pip3 install -e /src/local_lr
