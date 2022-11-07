import potentiometer_complex
import control_delta
import env_interaction
from identity_policy import IdentityLowLevelPolicy
import numpy as np
from rl_utils.analysis import reps_solve_info_analysis
from scipy.stats import norm

with open("./data/skill3_trained_1.pkl", "rb") as f:
    reps_policy_loaded = pickle.load(f)
    


for n, i in enumerate(pickles):
    with open(f'./triple_expt/{i}', 'rb') as handle:
        reps_policy_loaded = pickle.load(handle)

    verbose = True
    path_to_reps_info = i
    solve_env_info = reps_policy_loaded['solve_env_info']

    mean_param_hist = np.array(solve_env_info["history"]["policy_params_mean"])
    var_diag_param_hist = solve_env_info["history"]["policy_params_var_diag"]
    mean_param_hist, var_diag_param_hist
    
    var_diag_param_hist = np.vstack(var_diag_param_hist)

    x_axis = np.arange(-1, 1, 0.001)

    for j in range(4):
        print(j)
        for num, (m,v) in enumerate(zip(mean_param_hist[0][:,j], var_diag_param_hist[:,j])):
            norm_vals = np.array(norm.pdf(x_axis,m,v))
            if j%2==0:
    #             plt.ylim(0,1000)
#                 plt.plot((x_axis+1)*8.5, np.log((norm_vals+1)*8.5), color=colors[num])
                plt.plot((x_axis+1)*8.5, (norm_vals+1)*8.5
#                          , color=colors[num]
                        )
#                 plt.xlim(5,12.5)
            else:
#                 plt.plot((x_axis+1)*180, np.log((norm_vals+1)*180), color=colors[num])
                plt.plot((x_axis+1)*180, (norm_vals+1)*180, 
#                          color=colors[num]
                        )
#                 plt.xlim(100,300)
        plt.title(f"Robot: {n}, {var[j]}")
        plt.savefig(f'./triple_expt/Robot: {n}, {var[j]}.png', transparent=False,facecolor='white')
        plt.show()