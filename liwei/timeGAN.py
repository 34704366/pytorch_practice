from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN

print(ModelParameters)
# seq_len=24
# n_seq = 1
# hidden_dim=24
# gamma=1
#
# noise_dim = 32
# dim = 128
# batch_size = 128
#
# log_step = 100
# learning_rate = 5e-4
#
# stock_data = processed_stock(path='hushen300.csv', seq_len=seq_len)
# print(len(stock_data),stock_data[0].shape)
#
#
#
# gan_args = ModelParameters(batch_size=batch_size,
#                            lr=learning_rate,
#                            noise_dim=noise_dim,
#                            layers_dim=dim)
# synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=seq_len, n_seq=n_seq, gamma=1)
# synth.train(stock_data, train_steps=2000)
# synth.save('synthesizer_stock.pkl')
#
#
# synth_data = synth.sample(len(stock_data))
# print(synth_data.shape)

