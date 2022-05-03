from utils_general import *
from utils_methods import *

# Dataset initialization
###
[alpha, beta, theta, iid_sol, iid_data, name_prefix] = [0.0, 0.0, 0.0, True, True, 'syn_alpha-1_beta-1_theta0']

n_dim = 30
n_clnt = 10
n_cls = 2

data_obj = DatasetFD2(n_clnt, 0, "fd")

###
model_name = 'fd'  # Model type
com_amount = 500
save_period = 50
weight_decay = 1e-5
batch_size = 1024
act_prob = .1
lr_decay_per_round = 1
epoch = 10
learning_rate = 0.1
print_per = 5

# Model function
model_func = lambda: client_model(model_name, [n_dim, n_cls])
init_model = model_func()

"""
# Initalise the model for all methods
with torch.no_grad():
    init_model.fc.weight = torch.nn.parameter.Parameter(torch.zeros(n_cls, n_dim))
    init_model.fc.bias = torch.nn.parameter.Parameter(torch.zeros(n_cls))
"""


if not os.path.exists('Output/%s/' % (data_obj.name)):
    os.mkdir('Output/%s/' % (data_obj.name))

# Methods
####
print('FedDyn')

alpha_coef = 1e-2
[fed_mdls_sel_FedFyn, trn_perf_sel_FedFyn, tst_perf_sel_FedFyn,
 fed_mdls_all_FedFyn, trn_perf_all_FedFyn, tst_perf_all_FedFyn,
 fed_mdls_cld_FedFyn] = train_FedDyn(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                                     batch_size=batch_size,
                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                     model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
                                     save_period=save_period, lr_decay_per_round=lr_decay_per_round)
# Plot results
plt.figure(figsize=(6, 5))
plt.plot(np.arange(com_amount)+1, tst_perf_all_FedFyn[:,1], label='FedDyn')
plt.ylabel('Test Accuracy', fontsize=16)
plt.xlabel('Communication Rounds', fontsize=16)
plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
plt.grid()
plt.xlim([0, com_amount+1])
plt.title(data_obj.name, fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Output/%s/plot.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')
plt.show()
