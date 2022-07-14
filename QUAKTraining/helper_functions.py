from imports import *

def smoothen_integers(input_dist):
    if (input_dist%1==0).all():
        #print("integer distribution detected...")
        input_dist += np.random.normal(loc=0., scale=0.5, size=len(input_dist))
    return input_dist

def smoothen_padding(input_dist, padding_threshold=10, ref_neighbor=10):
    # padding_threshold: minimum number of entries that define a padded edge
    # ref_neighbor: distance to this neighbor defines width and position

    lower_edge = min(input_dist)
    n_lower_edge = sum(input_dist==lower_edge)
    lower_padding = n_lower_edge>padding_threshold
    upper_edge = max(input_dist)
    n_upper_edge = sum(input_dist==upper_edge)
    upper_padding = n_upper_edge>padding_threshold

    if not (lower_padding or upper_padding):
        print("no padding detected...")
        return input_dist
    elif not upper_padding:
        print("lower padding detected...")
    elif not lower_padding:
        print("upper padding detected...")
    else:
        print("upper and lower padding detected...")

    unique_vals = np.unique(input_dist)

    if lower_padding:
        near_neighbor_pos = unique_vals[1]
        ref_neighbor_pos = unique_vals[ref_neighbor]
        edge_width = abs(ref_neighbor_pos - near_neighbor_pos)

        gauss_vals = np.random.normal(loc=near_neighbor_pos-edge_width,
            scale=edge_width, size=n_lower_edge)
        input_dist[input_dist==lower_edge]=gauss_vals

    if upper_padding:
        near_neighbor_pos = unique_vals[-2]
        ref_neighbor_pos = unique_vals[-ref_neighbor-1]
        edge_width = abs(ref_neighbor_pos - near_neighbor_pos)

        gauss_vals = np.random.normal(loc=near_neighbor_pos+edge_width,
            scale=edge_width, size=n_upper_edge)
        input_dist[input_dist==upper_edge]=gauss_vals

    return input_dist

def get_batch_file(inp_sample_type, inp_batch_number,invert=False):
    from samples import sample_library_UL17, sample_library_UL18, bkg_loc, bkg_loc_old

    if 'bkg' in inp_sample_type.lower():
        if invert:
            nmax = len(os.listdir(bkg_loc)) - 1
            batch = bkg_loc+"BB_batch_{0}.h5".format(nmax - inp_batch_number)
        else:
            batch = bkg_loc+"BB_batch_{0}.h5".format(inp_batch_number)
    elif 'CMS' in inp_sample_type:
        if inp_batch_number < 100:
            batch = "/home/wmccorma/CASE_SmallDataset_April21/Data_2018A_%s.h5" % (inp_batch_number + 900)
        elif inp_batch_number < 110:
            batch = "/home/wmccorma/CASE_SmallDataset_April21/Data_2018A_%s.h5" % (inp_batch_number - 10)
        else:
            batch = "/home/wmccorma/CASE_SmallDataset_April21/Data_2018A_9.h5"
    else:
        samp_lib = None
        if "UL17" in inp_sample_type:
            samp_lib = sample_library_UL17
        if "UL18" in inp_sample_type:
            samp_lib = sample_library_UL18
        if samp_lib == None:
            print("ERROR: No sample found -- check that you put ULXX in the sample name. Check available samples in samples.py")
            batch = None
        else:
            batch = samp_lib[inp_sample_type]
    return batch

def extract_trainVars(sample,Mjj_cut=1200,pt_cut=550,eta_cut=None,add_mjj=False,add_lsf3=False):
    f = h5py.File(sample, "r")

    jet_kinematics = f['jet_kinematics']
    jet1_extraInfo = f['jet1_extraInfo']
    jet2_extraInfo = f['jet2_extraInfo']
    truth_label = f['truth_label']

    np.seterr(invalid = 'ignore')

    delta_eta = jet_kinematics[:,1]

    Mjj = np.reshape(jet_kinematics[:,0], (-1,1))
    Mj1 = np.reshape(jet_kinematics[:,5], (-1,1))
    Mj2 = np.reshape(jet_kinematics[:,9], (-1,1))

    jet1_pt = np.reshape(jet_kinematics[:,2], (-1,1))
    jet2_pt = np.reshape(jet_kinematics[:,6], (-1,1))

    jet1_tau1 = np.reshape(jet1_extraInfo[:,0], (-1,1))
    jet1_tau2 = np.reshape(jet1_extraInfo[:,1], (-1,1))
    jet1_tau3 = np.reshape(jet1_extraInfo[:,2], (-1,1))
    jet1_tau4 = np.reshape(jet1_extraInfo[:,3], (-1,1))
    jet1_lsf3 = np.reshape(jet1_extraInfo[:,4], (-1,1))
    jet1_btagscore = np.reshape(jet1_extraInfo[:,5],(-1,1))
    jet1_numpfconst = np.reshape(jet1_extraInfo[:,6],(-1,1))

    jet1_tau21 = jet1_tau2 / jet1_tau1
    jet1_tau32 = jet1_tau3 / jet1_tau2
    jet1_tau43 = jet1_tau4 / jet1_tau3
    jet1_sqrt_tau21 = np.sqrt(jet1_tau21) / jet1_tau1

    jet2_tau1 = np.reshape(jet2_extraInfo[:,0], (-1,1))
    jet2_tau2 = np.reshape(jet2_extraInfo[:,1], (-1,1))
    jet2_tau3 = np.reshape(jet2_extraInfo[:,2], (-1,1))
    jet2_tau4 = np.reshape(jet2_extraInfo[:,3], (-1,1))
    jet2_lsf3 = np.reshape(jet2_extraInfo[:,4], (-1,1))
    jet2_btagscore = np.reshape(jet2_extraInfo[:,5],(-1,1))
    jet2_numpfconst = np.reshape(jet2_extraInfo[:,6],(-1,1))

    jet2_tau21 = jet2_tau2 / jet2_tau1
    jet2_tau32 = jet2_tau3 / jet2_tau2
    jet2_tau43 = jet2_tau4 / jet2_tau3
    jet2_sqrt_tau21 = np.sqrt(jet2_tau21) / jet2_tau1

    all_vars = [Mj1, jet1_tau21, jet1_tau32, jet1_tau43, jet1_sqrt_tau21, jet1_btagscore, jet1_numpfconst,
                Mj2, jet2_tau21, jet2_tau32, jet2_tau43, jet2_sqrt_tau21, jet2_btagscore, jet2_numpfconst]
    varNames = [r'$M_{j1}$', r'Jet 1 $\tau_{21}$', r'Jet 1 $\tau_{32}$', r'Jet 1 $\tau_{43}$', r'Jet 1 $\tau_s$', r'Jet 1 $P_b$', r'Jet 1 $n_{pf}$',
                r'$M_{j2}$', r'Jet 2 $\tau_{21}$', r'Jet 2 $\tau_{32}$', r'Jet 2 $\tau_{43}$', r'Jet 2 $\tau_s$', r'Jet 2 $P_b$', r'Jet 2 $n_{pf}$']
    if add_mjj:
        all_vars.append(Mjj)
        varNames.append(r'$M_{jj}$')
    if add_lsf3:
        all_vars.append(jet1_lsf3)
        all_vars.append(jet2_lsf3)
        varNames.append(r'Jet 1 LSF$_3$')
        varNames.append(r'Jet 2 LSF$_3$')

    data = np.concatenate(all_vars, axis=1)

    indices = np.where((Mjj > Mjj_cut)
                              & (jet1_pt > pt_cut)
                              & (jet2_pt > pt_cut)
                              & (np.isfinite(jet1_tau21))
                              & (np.isfinite(jet1_tau32))
                              & (np.isfinite(jet1_tau43))
                              & (np.isfinite(jet1_sqrt_tau21))
                              & (np.isfinite(jet2_tau21))
                              & (np.isfinite(jet2_tau32))
                              & (np.isfinite(jet2_tau43))
                              & (np.isfinite(jet2_sqrt_tau21)))[0]

    if eta_cut is not None:
        eta_indices = np.where((np.abs(delta_eta) < eta_cut))[0]
        indices = np.intersect1d(indices, eta_indices)

    norm_data = data[indices]
    masses = Mjj[indices]

    return norm_data, masses, varNames

def LAPS_train(sample_type, num_batches=1, Mjj_cut=1200, pt_cut=550, eta_cut=None, inp_meanstd=None, invert=False, add_mjj=False, add_lsf3=False):
    #LAPS stands for Load And Process Samples
    norm_data = []
    for batch_number in range(num_batches):
        train_batch = get_batch_file(sample_type, batch_number, invert=invert)
        norm_data0, masses0, varNames = extract_trainVars(train_batch,Mjj_cut=Mjj_cut,pt_cut=pt_cut,eta_cut=eta_cut,add_mjj=add_mjj,add_lsf3=add_lsf3)
        if batch_number == 0:
            norm_data, masses = norm_data0, masses0
        else:
            norm_data = np.concatenate((norm_data,norm_data0),axis=0)
            masses = np.concatenate((masses,masses0),axis=0)

    smooth_jet1_numpfconst = smoothen_integers(norm_data[:,6])
    smooth_jet2_numpfconst = smoothen_integers(norm_data[:,13])

    unnorm_data = np.copy(norm_data)

    return norm_data, unnorm_data, masses, varNames

def train_NF(input_iterator,flowName,flow_type="NSQUAD",
             num_features=14,hidden_features=56,num_layers=4,num_blocks_per_layer=4,tail_bound=5.0,
             num_iter=1000,print_interval=20,patience=10,learning_rate=1e-3,save_model=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device =", device)

    loss_dict = dict()
    flow_list = []

    print("FCNN Hidden Layer Width: ", hidden_features)

    print('------------------------------------')

    base_dist = StandardNormal(shape=[num_features])

    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=num_features))
        if flow_type == 'MAF':
            transforms.append(MaskedAffineAutoregressiveTransform(features=num_features,
                                                              hidden_features=hidden_features))
        elif flow_type == 'NSQUAD':
            transforms.append(MaskedPiecewiseQuadraticAutoregressiveTransform(features=num_features,
                                                              hidden_features=hidden_features, tail_bound = tail_bound, tails='linear'))
        elif flow_type == 'NSRATQUAD':
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=num_features,
                                                              hidden_features=hidden_features, tail_bound = tail_bound, tails='linear'))

    transform = CompositeTransform(transforms)

    flow = Flow(transform, base_dist)

    optimizer = optim.Adam(flow.parameters(),lr=learning_rate)

    tick = time.time()

    min_loss = 999999
    best_flow = None

    cur_losses = []
    patience_count = 0

    for i in range(num_iter):
        terminate = False

        for batch_idx, x in enumerate(input_iterator):

            optimizer.zero_grad()
            loss = -flow.log_prob(inputs=x)[0].mean()
            loss.backward()
            optimizer.step()

            if batch_idx == len(input_iterator) - 1 :
                if (i + 1) % print_interval == 0:
                    print('Iteration {} Complete'.format(i + 1))

                print_loss = loss.detach().cpu().numpy()
                cur_losses.append(print_loss)
                print('Loss: ', print_loss)

                if print_loss < min_loss:
                    patience_count = 0
                    best_flow = flow
                    if save_model:
                        torch.save(flow, "saved_flows/{0}".format(flowName))
                    min_loss = print_loss
                    if (i + 1) % print_interval == 0:
                        print('SAVING MODEL')
                else:
                    patience_count += 1
                    if (i + 1) % print_interval == 0:
                        print('NOT SAVING MODEL (PATIENCE = %s)' % patience_count)
                    if patience_count == patience:
                        terminate = True
                        break

                tock = time.time()

                if (i + 1) % print_interval == 0:
                    print('Time: ', tock - tick)
                    print('------------------------------------')
                    #!nvidia-smi

        if terminate:
            break

    flow_list.append(best_flow)

    print('------------------------------------')

    return best_flow, min_loss, cur_losses

def plot_input_variables(data,sample,varNames):
    num_features = int(data.shape[1])
    plt.figure(figsize=(18,18))
    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.suptitle("Training Inputs : {0}".format(sample),fontsize=16,y=0.93)
    for index in range(num_features):
        plt.subplot(4,4,index+1)
        n, bins, patches = plt.hist(data[:, index], bins=50, histtype='step', label='Truth')
        if index % 16 == 0:
            plt.legend(loc='best')
        plt.title(varNames[index])
    if not os.path.isdir("plots/training_plots/"+sample):
        os.mkdir("plots/training_plots/"+sample)
    plt.savefig("plots/training_plots/"+sample+"/train_input.pdf")
    plt.close()

def plot_density_estimation(flow,data,varNames,pdfpage=None):
    n_samples = int(data.shape[0])
    num_features = int(data.shape[1])
    with torch.no_grad():
        samples = flow.sample(n_samples).detach().cpu().numpy()

    plt.figure(figsize=(18,18))
    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.suptitle("Training samples vs truth",fontsize=16,y=0.93)
    for index in range(num_features):
        plt.subplot(4,4,index+1)
        n, bins, patches = plt.hist(data[:, index], bins=50, histtype='step', label='Truth')
        plt.hist(samples[:, index], bins=bins, histtype='step', label='NF Density')
        if index % 16 == 0:
            plt.legend(loc='best')
        plt.title(varNames[index])
    #plt.savefig(plot_dir+"bkg_train_sample_densities.pdf")
    #plt.show()
    if pdfpage is not None:
        pdfpage.savefig()
        plt.close()

def make_aucs_bkgtr(bkg_flow,sig_data,bkg_data,save=False,pdfpage=None):
    bkgtr_bkgloss = -bkg_flow.eval_log_prob(bkg_data)[0]
    bkgtr_sigloss = -bkg_flow.eval_log_prob(sig_data)[0]
    bins = np.linspace(0,100,10001)
    tpr = []
    fpr = []
    for cut in bins:
        tpr.append(np.where(bkgtr_sigloss>cut)[0].shape[0]/len(bkgtr_sigloss))
        fpr.append(np.where(bkgtr_bkgloss>cut)[0].shape[0]/len(bkgtr_bkgloss))

    tpr = np.array(tpr)
    fpr = np.array(fpr)
    nonzero_idx = np.nonzero(fpr)
    tpr_inverse = tpr[nonzero_idx]
    fpr_inverse = 1/fpr[nonzero_idx]
    bkgtr_auc = metrics.auc(fpr,tpr)

    plt.figure(figsize=(16,8))

    ax1 = plt.subplot(121)
    plt.plot(tpr_inverse,fpr_inverse)
    plt.xlabel(r'$\epsilon_{sig}$',fontsize=15)
    plt.ylabel(r'$1/\epsilon_{bkg}$',fontsize=15)
    plt.yscale('log')
    plt.title('Bkg-trained Pure NF Model')

    ax2 = plt.subplot(122)
    plt.plot(fpr,tpr)
    plt.xlabel(r'$\epsilon_{bkg}$',fontsize=15)
    plt.ylabel(r'$\epsilon_{sig}$',fontsize=15)
    plt.title('Bkg-trained Pure NF Model (AUC = %s)' % round(bkgtr_auc,3))

    if pdfpage is not None:
        pdfpage.savefig()
        plt.close()

    return

def make_aucs_sigtr(sig_flow,sig_data,bkg_data,save=False,pdfpage=None):
    sigtr_bkgloss = -sig_flow.eval_log_prob(bkg_data)[0]
    sigtr_sigloss = -sig_flow.eval_log_prob(sig_data)[0]
    bins = np.linspace(0,100,10001)
    tpr = []
    fpr = []
    for cut in bins:
        tpr.append(np.where(sigtr_sigloss<cut)[0].shape[0]/len(sigtr_sigloss))
        fpr.append(np.where(sigtr_bkgloss<cut)[0].shape[0]/len(sigtr_bkgloss))

    tpr = np.array(tpr)
    fpr = np.array(fpr)
    nonzero_idx = np.nonzero(fpr)
    tpr_inverse = tpr[nonzero_idx]
    fpr_inverse = 1/fpr[nonzero_idx]
    sigtr_auc = metrics.auc(fpr,tpr)

    plt.figure(figsize=(16,8))

    ax1 = plt.subplot(121)
    plt.plot(tpr_inverse,fpr_inverse)
    plt.xlabel(r'$\epsilon_{sig}$',fontsize=15)
    plt.ylabel(r'$1/\epsilon_{bkg}$',fontsize=15)
    plt.yscale('log')
    plt.title('Sig-trained Pure NF Model')

    ax2 = plt.subplot(122)
    plt.plot(fpr,tpr)
    plt.xlabel(r'$\epsilon_{bkg}$',fontsize=15)
    plt.ylabel(r'$\epsilon_{sig}$',fontsize=15)
    plt.title('Sig-trained Pure NF Model (AUC = %s)' % round(sigtr_auc,3))

    if pdfpage is not None:
        pdfpage.savefig()
        plt.close()

    return

def draw_2d_quak_space(sig_flow,bkg_flow,data,title,pdfpage=None):
    bkg_loss = -bkg_flow.eval_log_prob(data)[0]
    sig_loss = -sig_flow.eval_log_prob(data)[0]

    plt.figure(figsize=(10,10))
    bins = np.linspace(0,50,51)
    h, x_edge, y_edge, _ = plt.hist2d(bkg_loss,sig_loss,cmap=plt.cm.jet,bins=bins)
    plt.colorbar()
    plt.xlabel("BKG-Trained Model Loss")
    plt.ylabel("SIG-Trained Model Loss")
    plt.title(title)

    if pdfpage is not None:
        pdfpage.savefig()
        plt.close()

    return

def draw_2d_quak_SigVsBkg(sig_flow,bkg_flow,sig_data,bkg_data,xlim=50,ylim=50,pdfpage=None):
    sigtr_sigloss = -sig_flow.eval_log_prob(sig_data)[0]
    sigtr_bkgloss = -sig_flow.eval_log_prob(bkg_data)[0]

    bkgtr_sigloss = -bkg_flow.eval_log_prob(sig_data)[0]
    bkgtr_bkgloss = -bkg_flow.eval_log_prob(bkg_data)[0]

    plt.figure(figsize=(10,10))
    plt.scatter(bkgtr_bkgloss,sigtr_bkgloss,s=2,label = "Bkg Test Data")
    plt.scatter(bkgtr_sigloss,sigtr_sigloss,s=2,label="Sig Test Data")
    plt.xlim([0,xlim])
    plt.ylim([0,ylim])
    plt.title("Signal & Bkg in 2D QUAK Space")
    plt.legend(loc='upper right',fontsize=14)

    if pdfpage is not None:
        pdfpage.savefig()
        plt.close()

    return

def draw_2d_quak_MultiSigVsBkg(sig_flow,bkg_flow,sig_data,sig_labels,bkg_data,xlim=50,ylim=50,pdfpage=None):
    bkgtr_bkgloss = -bkg_flow.eval_log_prob(bkg_data)[0]
    sigtr_bkgloss = -sig_flow.eval_log_prob(bkg_data)[0]

    plt.figure(figsize=(10,10))
    plt.scatter(bkgtr_bkgloss,sigtr_bkgloss,s=2,label = "Bkg Test Data")
    for i,dataset in enumerate(sig_data):
        sigtr_sigloss = -sig_flow.eval_log_prob(dataset)[0]
        bkgtr_sigloss = -bkg_flow.eval_log_prob(dataset)[0]
        plt.scatter(bkgtr_sigloss,sigtr_sigloss,s=2,label=sig_labels[i])
    plt.xlim([0,xlim])
    plt.ylim([0,ylim])
    plt.title("Signal & Bkg in 2D QUAK Space")
    plt.legend(loc='upper right',fontsize=14)

    if pdfpage is not None:
        pdfpage.savefig()
        plt.close()

    return

def make_summaryPdf(name,sig_flow,bkg_flow,sig_train,bkg_train,sig_test,bkg_test,varNames):
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages('plots/training_plots/'+name+'/summary.pdf') as pdf:
        plot_density_estimation(sig_flow,sig_train,varNames,pdfpage=pdf)
        plot_density_estimation(bkg_flow,bkg_train,varNames,pdfpage=pdf)
        make_aucs_bkgtr(bkg_flow,sig_test,bkg_test,pdfpage=pdf)
        make_aucs_sigtr(sig_flow,sig_test,bkg_test,pdfpage=pdf)
        draw_2d_quak_space(sig_flow,bkg_flow,bkg_test,"Background Test Data",pdfpage=pdf)
        draw_2d_quak_space(sig_flow,bkg_flow,sig_test,"Signal Test Data",pdfpage=pdf)
        draw_2d_quak_SigVsBkg(sig_flow,bkg_flow,sig_test,bkg_test,pdfpage=pdf)

def clip_tails(data,bound=10.0):
    nev_init = data.shape[0]
    good = np.all(data < bound, axis=1)
    out = data[good]
    nev_fin = out.shape[0]
    print("Initial: {0} events \nFinal: {1} events \nRemoved {2:.4f}%".format(nev_init,nev_fin,100*(nev_init-nev_fin)/nev_init))
    return out

def prepare_stats_data(sig_flow,bkg_flow,sig_test,bkg_test):
    sig_data, sig_unnorm_data, sig_masses = sig_test
    bkg_data, bkg_unnorm_data, bkg_masses = bkg_test

    ns = sig_data.shape[0]
    nb = bkg_data.shape[0]

    print("Using {0} Signal Events".format(ns))
    print("Using {0} Bkg Events".format(nb))
    print("{0:.4f}% signal injection".format(100*ns/(ns+nb)))

    bkg_data_WL = np.concatenate((bkg_data, np.zeros((nb,1), dtype='float32')), axis=1)
    sig_data_WL = np.concatenate((sig_data, np.ones((ns,1), dtype='float32')), axis=1)

    test_data = np.concatenate((bkg_data_WL, sig_data_WL), axis=0)
    test_masses = np.concatenate((bkg_masses, sig_masses), axis=0)

    test_data_WM = np.concatenate((test_masses, test_data), axis=1)

    np.random.shuffle(test_data_WM)

    test_masses = test_data_WM[:,0]
    test_labels = test_data_WM[:,-1]
    test_data = test_data_WM[:,1:-1]

    # calculating losses
    bkg_loss_indices = np.where(test_labels==0)
    sig_loss_indices = np.where(test_labels==1)

    bkgtr_test_losses = -bkg_flow.eval_log_prob(test_data)[0]
    sigtr_test_losses = -sig_flow.eval_log_prob(test_data)[0]

    return test_masses, sigtr_test_losses, bkgtr_test_losses, test_labels

def normalize_data(data,inp_mean=None,inp_std=None):
    if inp_mean is None or inp_std is None:
        for index in range(data.shape[1]):
            mean = np.mean(data[:,index])
            std = np.std(data[:,index])
            data[:,index] = (data[:,index] - mean)/std
        return data
    else:
        for index in range(data.shape[1]):
            mean = inp_mean[index]
            std = inp_std[index]
            data[:,index] = (data[:,index]-mean)/std
        return data

def load_full(sample,num_batches=1,bkg_mean=None,bkg_std=None,add_mjj=False,add_lsf3=False):
    data, unnorm_data, masses, varNames = LAPS_train(sample,num_batches=num_batches,add_mjj=add_mjj,add_lsf3=add_lsf3)
    if bkg_mean is not None and bkg_std is not None:
        data = normalize_data(data,inp_mean=bkg_mean,inp_std=bkg_std)
    else:
        data = normalize_data(data)
    return data, unnorm_data, masses

def load_bkg_batch(sample,batch,Mjj_cut=1200,pt_cut=550,eta_cut=None,add_mjj=False,add_lsf3=False):
    train_batch = get_batch_file(sample, batch)
    norm_data, masses, varNames = extract_trainVars(train_batch,Mjj_cut=Mjj_cut,pt_cut=pt_cut,eta_cut=eta_cut,add_mjj=add_mjj,add_lsf3=add_lsf3)

    smooth_jet1_numpfconst = smoothen_integers(norm_data[:,6])
    smooth_jet2_numpfconst = smoothen_integers(norm_data[:,13])

    unnorm_data = np.copy(norm_data)
    norm_data = normalize_data(norm_data)

    return norm_data, unnorm_data, masses

def load_and_split_bkg(sample,num_batches=1,add_mjj=False,add_lsf3=False):
    # load data
    bkg_train, bkg_train_unnorm, bkg_train_masses, varNames = LAPS_train(sample_type=sample, num_batches=num_batches, add_mjj=add_mjj, add_lsf3=add_lsf3)
    bkg_test, bkg_test_unnorm, bkg_test_masses, varNames = LAPS_train(sample_type=sample, num_batches=num_batches, invert=True, add_mjj=add_mjj, add_lsf3=add_lsf3)
    # normalize to mean 0, std 1
    bkg_train = normalize_data(bkg_train)
    bkg_test = normalize_data(bkg_test)
    # get means & stds for unnormalized data
    bkg_mean_train = np.mean(bkg_train_unnorm, axis=0)
    bkg_std_train = np.std(bkg_train_unnorm, axis=0)
    bkg_mean_test = np.mean(bkg_test_unnorm, axis=0)
    bkg_std_test = np.std(bkg_test_unnorm, axis=0)
    n_bkg_train = bkg_train_masses.size
    n_bkg_test = bkg_test_masses.size
    print("{0} {1} training events".format(n_bkg_train,sample))
    print("{0} {1} testing events".format(n_bkg_test,sample))

    return [(bkg_train, bkg_train_unnorm, bkg_train_masses),
            (bkg_test, bkg_test_unnorm, bkg_test_masses),
            (bkg_mean_train, bkg_std_train), (bkg_mean_test, bkg_std_test),
            varNames]

def load_and_split_sig(sample,train_frac,bkg_mean_train,bkg_std_train,bkg_mean_test,bkg_std_test,num_batches=1,add_mjj=False,add_lsf3=False):
    # load data
    sig_data, sig_unnorm_data, sig_masses, varNames = LAPS_train(sample, num_batches = 1, add_mjj=add_mjj, add_lsf3=add_lsf3)
    # split manually into train/test
    n_sig = sig_masses.size
    n_sig_train = int(train_frac*n_sig)
    sig_train, sig_train_unnorm, sig_train_masses = sig_data[:n_sig_train], sig_unnorm_data[:n_sig_train], sig_masses[:n_sig_train]
    sig_test, sig_test_unnorm, sig_test_masses = sig_data[n_sig_train:], sig_unnorm_data[n_sig_train:], sig_masses[n_sig_train:]
    # normalize signal to bkg mean/std
    sig_train = normalize_data(sig_train,inp_mean=bkg_mean_train,inp_std=bkg_std_train)
    sig_test = normalize_data(sig_test,inp_mean=bkg_mean_test,inp_std=bkg_std_test)
    # print info
    n_sig_train = sig_train_masses.size
    n_sig_test = sig_test_masses.size
    print("{0} {1} training events".format(n_sig_train,sample))
    print("{0} {1} testing events".format(n_sig_test,sample))

    return [(sig_train, sig_train_unnorm, sig_train_masses),
            (sig_test, sig_test_unnorm, sig_test_masses),
            varNames]

def load_test_sig(sample,train_frac,bkg_mean_test,bkg_std_test,num_batches=1,add_mjj=False,add_lsf3=False):
    # load data
    sig_data, sig_unnorm_data, sig_masses, varNames = LAPS_train(sample, num_batches = 1, add_mjj=add_mjj, add_lsf3=add_lsf3)
    # split manually into train/test
    n_sig = sig_masses.size
    n_sig_train = int(train_frac*n_sig)
    sig_train, sig_train_unnorm, sig_train_masses = sig_data[:n_sig_train], sig_unnorm_data[:n_sig_train], sig_masses[:n_sig_train]
    sig_test, sig_test_unnorm, sig_test_masses = sig_data[n_sig_train:], sig_unnorm_data[n_sig_train:], sig_masses[n_sig_train:]
    # normalize signal to bkg mean/std
    sig_test = normalize_data(sig_test,inp_mean=bkg_mean_test,inp_std=bkg_std_test)

    return sig_test, sig_test_unnorm, sig_test_masses, varNames

def load_and_split_multiSig(samples,train_frac,bkg_mean_train,bkg_std_train,bkg_mean_test,bkg_std_test,num_batches=1,add_mjj=False,add_lsf3=False):
    train_sets = []
    train_sets_unnorm = []
    train_sets_mass = []
    test_sets = []
    test_sets_unnorm = []
    test_sets_mass = []

    for sample in samples:
        data, unnorm_data, masses, varNames = LAPS_train(sample,num_batches=1,add_mjj=add_mjj, add_lsf3=add_lsf3)
        n_sig = masses.size
        n_train = int(train_frac*n_sig)
        train, train_unnorm, train_masses = data[:n_train], unnorm_data[:n_train], masses[:n_train]
        test, test_unnorm, test_masses = data[n_train:], unnorm_data[n_train:], masses[n_train:]
        train_sets.append(train)
        train_sets_unnorm.append(train_unnorm)
        train_sets_mass.append(train_masses)
        test_sets.append(test)
        test_sets_unnorm.append(test_unnorm)
        test_sets_mass.append(test_masses)
        print("{0} {1} training events".format(n_train,sample))
        print("{0} {1} testing events".format(n_sig-n_train,sample))

    sig_train = np.concatenate(train_sets,axis=0)
    sig_train_unnorm = np.concatenate(train_sets_unnorm,axis=0)
    sig_train_masses = np.concatenate(train_sets_mass,axis=0)
    sig_test = np.concatenate(test_sets,axis=0)
    sig_test_unnorm = np.concatenate(test_sets_unnorm,axis=0)
    sig_test_masses = np.concatenate(test_sets_mass,axis=0)

    sig_train = normalize_data(sig_train,inp_mean=bkg_mean_train,inp_std=bkg_std_train)
    sig_test = normalize_data(sig_test,inp_mean=bkg_mean_test,inp_std=bkg_std_test)
    # print info
    n_sig_train = sig_train_masses.size
    n_sig_test = sig_test_masses.size

    return [(sig_train, sig_train_unnorm, sig_train_masses),
            (sig_test, sig_test_unnorm, sig_test_masses),
            varNames]

def train_pipeline_bkg(bkg_sample,params_dict):
    # parse model parameters
    clip = params_dict["clip"] if "clip" in params_dict.keys() else None
    flow_type = params_dict["flow_type"] if "flow_type" in params_dict.keys() else "NSRATQUAD"
    tail_bound = params_dict["tail_bound"] if "tail_bound" in params_dict.keys() else 10
    hidden_features = params_dict["hidden_features"] if "hidden_features" in params_dict.keys() else 120
    num_layers = params_dict["num_layers"] if "num_layers" in params_dict.keys() else 6
    num_blocks_per_layer = params_dict["num_blocks_per_layer"] if "num_blocks_per_layer" in params_dict.keys() else 4
    patience = params_dict["patience"] if "patience" in params_dict.keys() else 20
    learning_rate = params_dict["learning_rate"] if "learning_rate" in params_dict.keys() else 5e-5
    save_model = params_dict["save_model"] if "save_model" in params_dict.keys() else False
    bs = params_dict["bs"] if "bs" in params_dict.keys() else 10000
    add_mjj = params_dict["add_mjj"] if "add_mjj" in params_dict.keys() else False
    add_lsf3 = params_dict["add_lsf3"] if "add_lsf3" in params_dict.keys() else False

    # loading in bkg datasets
    bkg_info = load_and_split_bkg(bkg_sample,add_mjj=add_mjj,add_lsf3=add_lsf3)
    bkg_train, bkg_train_unnorm, bkg_train_masses = bkg_info[0]
    bkg_test, bkg_test_unnorm, bkg_test_masses = bkg_info[1]
    bkg_mean_train, bkg_std_train = bkg_info[2]
    bkg_mean_test, bkg_std_test = bkg_info[3]
    varNames = bkg_info[4]
    num_features = bkg_train.shape[1]

    # clipping long tails
    if clip is not None:
        bkg_train = clip_tails(bkg_train,bound=clip)

    # plotting input variables
    flowName = '{0}_clip{1}_{2}_k{3}_hf{4}_nbpl{5}_tb{6}'.format(bkg_sample, clip, flow_type, num_layers, hidden_features, num_blocks_per_layer, tail_bound)
    if add_mjj:
        flowName += "_addMjj"
    if add_lsf3:
        flowName += "_addLSF3"
    plot_input_variables(bkg_train,flowName,varNames)

    # creating tensors
    total_bkg = torch.tensor(bkg_train)
    # creating iterators
    bkg_train_iterator = utils.DataLoader(total_bkg,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))

    # train the model
    bkgFlowName = flowName+".pt"
    bkg_flow, min_loss, bkg_losses = train_NF(bkg_train_iterator,bkgFlowName,flow_type=flow_type,num_features=num_features,
                                                  tail_bound=tail_bound,hidden_features=hidden_features,num_layers=num_layers,
                                                  num_blocks_per_layer=num_blocks_per_layer,
                                                  learning_rate=learning_rate,patience=patience,
                                                  save_model=save_model)
    # plot training curve
    plt.figure(figsize=(8,6))
    plt.plot(bkg_losses)
    plt.xlabel("Epoch")
    plt.ylabel("{0} training loss".format(bkg_sample))
    plt.savefig("plots/training_plots/"+flowName+"/training_curve.pdf")

    # saving training data
    with open("plots/training_plots/"+flowName+"/train_info.txt","w") as f:
        f.write("Loss: {0}\n".format(min_loss))
        f.write("Params: ")
        f.write(json.dumps(params_dict))

    return bkg_flow

def train_pipeline_sig(sig_sample,bkg_sample,params_dict,train_frac=0.9,bkg_model=None):
    # parse model parameters
    clip = params_dict["clip"] if "clip" in params_dict.keys() else None
    flow_type = params_dict["flow_type"] if "flow_type" in params_dict.keys() else "NSRATQUAD"
    tail_bound = params_dict["tail_bound"] if "tail_bound" in params_dict.keys() else 10
    hidden_features = params_dict["hidden_features"] if "hidden_features" in params_dict.keys() else 120
    num_layers = params_dict["num_layers"] if "num_layers" in params_dict.keys() else 6
    num_blocks_per_layer = params_dict["num_blocks_per_layer"] if "num_blocks_per_layer" in params_dict.keys() else 4
    patience = params_dict["patience"] if "patience" in params_dict.keys() else 20
    learning_rate = params_dict["learning_rate"] if "learning_rate" in params_dict.keys() else 5e-5
    save_model = params_dict["save_model"] if "save_model" in params_dict.keys() else False
    bs = params_dict["bs"] if "bs" in params_dict.keys() else 10000
    add_mjj = params_dict["add_mjj"] if "add_mjj" in params_dict.keys() else False
    add_lsf3 = params_dict["add_lsf3"] if "add_lsf3" in params_dict.keys() else False

    # loading in bkg datasets
    bkg_info = load_and_split_bkg(bkg_sample,add_mjj=add_mjj,add_lsf3=add_lsf3)
    bkg_train, bkg_train_unnorm, bkg_train_masses = bkg_info[0]
    bkg_test, bkg_test_unnorm, bkg_test_masses = bkg_info[1]
    bkg_mean_train, bkg_std_train = bkg_info[2]
    bkg_mean_test, bkg_std_test = bkg_info[3]
    varNames = bkg_info[4]
    # loading signal datasets
    if type(sig_sample) == list:
        sig_info = load_and_split_multiSig(sig_sample,train_frac,
                                           bkg_mean_train,bkg_std_train,
                                           bkg_mean_test,bkg_std_test,
                                           add_mjj=add_mjj,add_lsf3=add_lsf3)
        sigName_format = '-and-'.join(sig_sample)
    else:
        sig_info = load_and_split_sig(sig_sample,train_frac,
                                      bkg_mean_train,bkg_std_train,
                                      bkg_mean_test,bkg_std_test,
                                      add_mjj=add_mjj,add_lsf3=add_lsf3)
        sigName_format = sig_sample
    sig_train_data, sig_train_unnorm_data, sig_train_masses = sig_info[0]
    sig_test_data, sig_test_unnorm_data, sig_test_masses = sig_info[1]
    varNames = sig_info[2]
    num_features = sig_train_data.shape[1]

    # clipping long tails
    if clip is not None:
        sig_train_data = clip_tails(sig_train_data,bound=clip)

    # plotting input data
    flowName = '{0}_clip{1}_{2}_k{3}_hf{4}_nbpl{5}_tb{6}'.format(sigName_format, clip, flow_type, num_layers, hidden_features, num_blocks_per_layer, tail_bound)
    if add_mjj:
        flowName += "_addMjj"
    if add_lsf3:
        flowName += "_addLSF3"
    plot_input_variables(sig_train_data,flowName,varNames)

    # creating tensors
    total_sig = torch.tensor(sig_train_data)
    # creating iterators
    sig_train_iterator = utils.DataLoader(total_sig,batch_size=bs,shuffle=True,generator=torch.Generator(device='cuda'))

    # train the model
    sigFlowName = flowName+".pt"
    sig_flow, min_loss, sig_losses = train_NF(sig_train_iterator,sigFlowName,flow_type=flow_type,num_features=num_features,
                                                  tail_bound=tail_bound,hidden_features=hidden_features,num_layers=num_layers,
                                                  num_blocks_per_layer=num_blocks_per_layer,
                                                  learning_rate=learning_rate,patience=patience,
                                                  save_model=save_model)
    # plot training curve
    plt.figure(figsize=(8,6))
    plt.plot(sig_losses)
    plt.xlabel("Epoch")
    plt.ylabel("{0} training loss".format(sigName_format))
    plt.savefig("plots/training_plots/"+flowName+"/training_curve.pdf")

    # saving training data
    with open("plots/training_plots/"+flowName+"/train_info.txt","w") as f:
        f.write("Loss: {0}\n".format(min_loss))
        f.write("Params: ")
        f.write(json.dumps(params_dict))

    # make pdf summary with background comoparisons
    bkg_flow = load_model(bkg_sample,params=params_dict,name=bkg_model)
    if bkg_flow is not None:
        make_summaryPdf(flowName,sig_flow,bkg_flow,sig_train_data,bkg_train,sig_test_data,bkg_test,varNames)

    sig_train_suite = (sig_train_data, sig_train_unnorm_data, sig_train_masses)
    sig_test_suite = (sig_test_data, sig_test_unnorm_data, sig_test_masses)
    bkg_test_suite = (bkg_test, bkg_test_unnorm, bkg_test_masses)

    return [sig_flow, bkg_flow, sig_train_suite, sig_test_suite, bkg_test_suite]

def load_model(sample="",params={},name=None,device=torch.device("cuda:0")):
    if name is not None:
        if not os.path.exists("saved_flows/"+name):
            print("ERROR: no trained model exists!")
            return None
        else:
            flow = torch.load("saved_flows/"+name,map_location=device)
            return flow
    else:
        # parse model parameters
        clip = params_dict["clip"] if "clip" in params_dict.keys() else None
        flow_type = params_dict["flow_type"] if "flow_type" in params_dict.keys() else "NSRATQUAD"
        tail_bound = params_dict["tail_bound"] if "tail_bound" in params_dict.keys() else 10
        hidden_features = params_dict["hidden_features"] if "hidden_features" in params_dict.keys() else 120
        num_layers = params_dict["num_layers"] if "num_layers" in params_dict.keys() else 6
        num_blocks_per_layer = params_dict["num_blocks_per_layer"] if "num_blocks_per_layer" in params_dict.keys() else 4
        patience = params_dict["patience"] if "patience" in params_dict.keys() else 20
        learning_rate = params_dict["learning_rate"] if "learning_rate" in params_dict.keys() else 5e-5
        save_model = params_dict["save_model"] if "save_model" in params_dict.keys() else False
        bs = params_dict["bs"] if "bs" in params_dict.keys() else 10000
        add_mjj = params_dict["add_mjj"] if "add_mjj" in params_dict.keys() else False
        add_lsf3 = params_dict["add_lsf3"] if "add_lsf3" in params_dict.keys() else False

        # looking for pre-trained model
        flowName = '{0}_clip{1}_{2}_k{3}_hf{4}_nbpl{5}_tb{6}'.format(sample, clip, flow_type, num_layers, hidden_features, num_blocks_per_layer, tail_bound)
        if add_mjj:
            flowName += "_addMjj"
        if add_lsf3:
            flowName += "_addLSF3"
        flowName += ".pt"

        if not os.path.exists("saved_flows/"+flowName):
            print("ERROR: no trained model exists!")
            return None
        else:
            flow = torch.load("saved_flows/"+flowName,map_location=device)
            return flow

def evalAndSave(sig_flow,sig_flow_name,bkg_flow,bkg_flow_name,test_data,test_masses,test_name,label,transform=None,outdir="test_data_forStats"):
    nev = test_data.shape[0]
    labels = label*np.ones(nev,dtype='float32')
    bkgtr_test_losses = -bkg_flow.eval_log_prob(test_data)[0]
    sigtr_test_losses = -sig_flow.eval_log_prob(test_data)[0]
    
    if transform is not None:
        combined_data = np.concatenate((test_masses[:,np.newaxis],sigtr_test_losses[:,np.newaxis],bkgtr_test_losses[:,np.newaxis]),axis=1)
        transformed = transform.transform(combined_data)
        sigtr_test_losses = transformed[:,1]
        bkgtr_test_losses = transformed[:,2]
    
    output = np.array([test_masses,sigtr_test_losses,bkgtr_test_losses,labels])

    if not os.path.isdir(outdir+"/sigTrain{0}_bkgTrain{1}".format(sig_flow_name,bkg_flow_name)):
        os.mkdir(outdir+"/sigTrain{0}_bkgTrain{1}".format(sig_flow_name,bkg_flow_name))
    outFile = "sigTrain{0}_bkgTrain{1}/eval_{2}.npy".format(sig_flow_name,bkg_flow_name,test_name)
    np.save(outdir+"/"+outFile,output)

    del bkgtr_test_losses, sigtr_test_losses
    del output
    torch.cuda.empty_cache()

def sig_vs_bkg_2DQuakSpace(sig_train_name,bkg_train_name,sig_bkgtr_test_losses,sig_sigtr_test_losses,bkg_bkgtr_test_losses,bkg_sigtr_test_losses,sig_name,mjjDecorr=False,add_lsf3=False):
    plt.figure(figsize = (10,10))
    plt.scatter(bkg_bkgtr_test_losses,bkg_sigtr_test_losses,s=2,label="{0}".format(bkg_train_name))
    plt.scatter(sig_bkgtr_test_losses,sig_sigtr_test_losses,s=2,label="{0}".format(sig_name))
    plt.legend(loc='upper right',fontsize=14)
    if mjjDecorr:
        plt.xlim([-20,20])
        plt.ylim([-20,20])
        plt.xticks(np.arange(-20,25,step=5))
        plt.yticks(np.arange(-20,25,step=5))
    else:
        plt.xlim([0,50])
        plt.ylim([0,50])
        plt.xticks(np.arange(0,55,step=5))
        plt.yticks(np.arange(0,55,step=5))
    plt.title("Signal Trained on {0}".format(sig_train_name))
    saveLoc = "plots/QUAK_spaces/sigTrain{0}_bkgTrain{1}".format(sig_train_name,bkg_train_name)
    if mjjDecorr:
        saveLoc +="_mjjDecorr"
    if add_lsf3:
        saveLoc += "_addLSF3"
    saveLoc += "/"
    if not os.path.isdir(saveLoc):
        os.mkdir(saveLoc)
    saveFile = saveLoc+"eval_{0}.png".format(sig_name)
    plt.savefig(saveFile)
    plt.close()

    del sig_bkgtr_test_losses, sig_sigtr_test_losses, bkg_bkgtr_test_losses, bkg_sigtr_test_losses

def marginal_log_prob(flow,inputs,axes,bounds=[-10,10],step=0.1):
    locs = np.arange(bounds[0],bounds[1],step=step) + step/2 # midpoints
    n_ax = len(axes)

    int_points = []
    def make_integral_points(locs,loop_depth,point):
        if loop_depth >= 1:
            for pt in locs:
                make_integral_points(locs,loop_depth-1,point+[pt])
        else:
            int_points.append(point)
    make_integral_points(locs,n_ax,[])

    total_prob = 0
    int_vol = step**n_ax
    for pt in int_points:
        int_inputs = np.copy(inputs)
        for i,ax in enumerate(axes):
            int_inputs[:,ax] = pt[i]
        tens_inputs = torch.as_tensor(int_inputs)
        total_prob += np.exp(flow.eval_log_prob(tens_inputs)[0])*int_vol
        del int_inputs, tens_inputs

    return np.log(total_prob)

def integrated_likelihood(flow,data,axis,bounds=[-10,10],step=0.1):
    grid = np.arange(bounds[0],bounds[1],step=step) + step/2
    num_pts = data.shape[0]
    data_grid = np.tile(data,[len(grid),1])
    data_grid[:,axis] = np.repeat(grid,num_pts)
    num_splits = data_grid.shape[0] // 100000 if data_grid.shape[0] > 100000 else 1
    losses = []
    t1 = time.time()
    for subarr in np.array_split(data_grid,num_splits):
        loss = flow.eval_log_prob(subarr)[0]
        losses.append(loss)
    t2 = time.time()
    print("took {0:.4f} secs to process {1} events".format(t2-t1,data_grid.shape[0]))
    losses = np.concatenate(losses)
    losses_by_pt = []
    for i in range(num_pts):
        losses_by_pt.append(step*np.sum(np.exp(losses[i::num_pts])))
    return np.array(losses_by_pt)

# Neural network to evaluate models
class simpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(2,128),
            nn.Sigmoid(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        return self.transform(x)
    
def NNmodelPerformance(sig_trainName,sig_testName,bkg_name="QCDBKG",base_dir="data_forStats_mjjDecorrelate/",
                       n_train=10000,n_epoch=1000):
    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = base_dir+"sigTrain{0}_bkgTrain{1}/".format(sig_trainName,bkg_name)
    
    # loading bkg data
    bkg_samples = [f for f in os.listdir(data_dir) if bkg_name in f]
    bkg_sigtr_losses = []
    bkg_bkgtr_losses = []
    for samp in bkg_samples:
        arr = np.load(data_dir+samp)
        bkg_sigtr_losses.append(arr[1])
        bkg_bkgtr_losses.append(arr[2])
        del arr
    bkg_sigtr_losses = np.concatenate(bkg_sigtr_losses)
    bkg_bkgtr_losses = np.concatenate(bkg_bkgtr_losses)
    bkg_data = np.column_stack((bkg_bkgtr_losses,bkg_sigtr_losses))
    del bkg_sigtr_losses, bkg_bkgtr_losses
    
    # signal data
    arr = np.load(data_dir+"eval_{0}.npy".format(sig_testName))
    sig_sigtr_losses = arr[1]
    sig_bkgtr_losses = arr[2]
    sig_data = np.column_stack((sig_bkgtr_losses,sig_sigtr_losses))
    del arr, sig_sigtr_losses, sig_bkgtr_losses
    
    # train/test split
    sig_train = sig_data[:n_train]
    sig_test = sig_data[n_train:]
    bkg_train = bkg_data[:n_train]
    bkg_test = bkg_data[n_train:]
    
    train_losses = np.concatenate((sig_train,bkg_train),axis=0)
    train_labels = np.concatenate((np.ones(sig_train.shape[0]),np.zeros(bkg_train.shape[0])))
    train = np.concatenate((train_losses,train_labels[:,np.newaxis]),axis=1)
    np.random.shuffle(train)
    train = torch.tensor(train).to(device)

    test_losses = np.concatenate((sig_test,bkg_test),axis=0)
    test_labels = np.concatenate((np.ones(sig_test.shape[0]),np.zeros(bkg_test.shape[0])))
    test = np.concatenate((train_losses,train_labels[:,np.newaxis]),axis=1)
    np.random.shuffle(test)

    test = torch.tensor(test).to(device)
    
    # Build & train model
    model = simpleNN()
    model = model.double()
    model = model.cuda()
    
    optimizer = optim.Adam(model.parameters())
    f_loss = nn.MSELoss()

    n_epoch = 1000
    losses = []
    for i in range(n_epoch):
        x, y = train[:,:2], train[:,2:]
        output = model(x)
        loss = f_loss(output,y)
        losses.append(loss.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("Epoch {}, Loss: {}".format(i,loss))
        
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(n_epoch),losses)
    plt.title("Sig train {}, bkg train {}, tested on {}".format(sig_trainName,bkg_name,sig_testName))
    plt.savefig(data_dir+"NN_loss_eval{}.pdf".format(sig_testName))
    plt.close()
    
    y_pred = model(test[:,:2]).detach().cpu().numpy().flatten()
    y_true = test[:,2:].cpu().numpy().flatten()
    
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    fpr,tpr,thresholds = roc_curve(y_true,y_pred)
    auc = roc_auc_score(y_true,y_pred)
    plt.figure(figsize=(8,8))
    plt.plot(fpr,tpr)
    plt.title("Sig train {}, bkg train {}, tested on {}".format(sig_trainName,bkg_name,sig_testName))
    plt.text(0.6,0.1,"AUC = {:.4f}".format(auc),fontsize=14)
    plt.savefig(data_dir+"NN_roc_eval{}.pdf".format(sig_testName))
    plt.close()

    del train, test, model
    
    return auc
    