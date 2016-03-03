import os
import numpy as np
from io_funcs.binary_io import BinaryIOCollection
from io_funcs.read_hts_label_file import readHTSlabelFile
from utils import data_normalization
from utils import eval_metrics

def modify_phone_labels(label_align_dir, out_dur_dir, out_mod_lab_dir, list_of_files):
    for i in range(len(list_of_files)):
        fname = list_of_files[i]
        print fname
        
        dur_arr = []
        ip1 = open(out_dur_dir+'/'+fname+'.dur', 'r')
        for j in ip1.readlines():
            fstr = j.strip().split()
            for k in range(5):
                dur_arr.append(fstr[k])
        ip1.close()
        
        ip2 = open(label_align_dir+'/'+fname+'.lab','r')
        op1 = open(out_mod_lab_dir+'/'+fname+'.lab','w')
        count=0
        prev_ed=0
        for j in ip2.readlines():
            fstr = j.strip().split()
            ftag = fstr[2]
            ph = ftag[ftag.index('-')+1:ftag.index('+')]
            if ph=='#':
                op1.write(str(prev_ed)+' '+str(prev_ed+int(fstr[1])-int(fstr[0]))+' '+fstr[2]+'\n')
                prev_ed = prev_ed+int(fstr[1])-int(fstr[0])
                continue;
            else:
                dr = dur_arr[count]
                dr = int(dr)*5*10000
                op1.write(str(prev_ed)+' '+str(prev_ed+dr)+' '+fstr[2]+'\n')
                prev_ed = prev_ed+dr
        
            count=count+1
    
        ip2.close()
        op1.close()

if __name__ == "__main__":
    
    htsclass = readHTSlabelFile()
    io_funcs = BinaryIOCollection()

    ### speaker ###
    speaker = 'blz16'
    decomposition_unit = 'phone'
    normalization = 'MVN'
    
    out_dim = 8
    CTC_classes = 12

    ### Absolute work path ###
    work_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Hybrid_prosody_model/'
    
    label_align_dir = os.path.join(work_dir, 'Data/inter-module/'+speaker+'/label_state_align')  
    feat_dir_path = 'dur_' + decomposition_unit + '_' + str(out_dim)
    #feat_dir_path = 'CTC_' + decomposition_unit + '_' + str(CTC_classes)
    out_dir = os.path.join(work_dir, 'Data/inter-module/'+speaker+'/duration_features/' + feat_dir_path + '/')
    
    out_dur_dir = os.path.join(work_dir, 'Data/inter-module/'+speaker+'/gen/dur/BLSTM_MDN_1024_6_LR_p001/')
    out_mod_lab_dir = os.path.join(work_dir, 'Data/inter-module/'+speaker+'/gen/modified_lab/BLSTM_MDN_1024_6_LR_p001/')
    
    DFP = True
    if DFP:
        
        extractDur = False
        perform_CTC_classes = False
        analysisDur = False
        denormDur  = True
        writeDur   = False
        modifyLabels = False
        skip_pause = True
        calcRMSE   = True
        
        if extractDur:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            filelist = os.path.join(work_dir, 'Data/fileList/'+speaker+'.scp')
            stat_fname = feat_dir_path + '.txt'
            stats_file = os.path.join(work_dir, 'Data/inter-module/'+speaker+'/misc/', stat_fname)
            list_of_files = io_funcs.load_file_list(filelist)
            
            all_files_ph = []
            dur_features = []; flens = []
            for i in range(len(list_of_files)):
                filename = list_of_files[i]
                print filename
                
                in_lab_file  = os.path.join(label_align_dir, filename + '.lab')
                [phone, st_arr, ph_arr, mean_f0_arr] = htsclass.read_state_align_label_file(in_lab_file)
                
                flens.append((len(phone)-2)*out_dim)
                for j in range(len(phone)):
                    if(phone[j] == '#'):
                        continue;
                    
                    all_files_ph.append(phone[j])
                    
                    ### phone duration ###
                    ph_start = int(ph_arr[0][j] / (np.power(10, 4) * 5));
                    ph_end = int(ph_arr[1][j] / (np.power(10, 4) * 5));
                    phone_num_of_frames = ph_end - ph_start;
                    
                    ### syllable duration ###
                    st_ph_indx = int(mean_f0_arr[j][2]);
                    fn_ph_indx = int(mean_f0_arr[j][3]);
                    if(st_ph_indx == 0 or fn_ph_indx == 0):
                        st_ph_indx = j
                        fn_ph_indx = j
                    ph_start = int(ph_arr[0][st_ph_indx] / (np.power(10, 4) * 5));
                    ph_end = int(ph_arr[1][fn_ph_indx] / (np.power(10, 4) * 5));
                    syl_num_of_frames = ph_end - ph_start;    
                    
                    ### word duration ###
                    st_ph_indx = int(mean_f0_arr[j][4]);
                    fn_ph_indx = int(mean_f0_arr[j][5]);
                    if(st_ph_indx == 0 or fn_ph_indx == 0):
                        st_ph_indx = j
                        fn_ph_indx = j
                    ph_start = int(ph_arr[0][st_ph_indx] / (np.power(10, 4) * 5));
                    ph_end = int(ph_arr[1][fn_ph_indx] / (np.power(10, 4) * 5));
                    word_num_of_frames = ph_end - ph_start;
                    
                    for k in range(5):
                        dur_features.append(st_arr[j][k]/(np.power(10,4)*5))
                    
                    dur_features.append(phone_num_of_frames)
                    dur_features.append(syl_num_of_frames)
                    dur_features.append(word_num_of_frames)
            
            ##### normalise the data #####
            print 'Normalising the data....'
            if(normalization == "MVN"):
                norm_data = data_normalization.MVN_normalize(dur_features, out_dim, stats_file)
            else:
                norm_data = dur_features
            
                    
            ##### write features into files ####
            print 'Writing features into output files...'
            count = 0;idx = 0;flength = 0;
            for k in range(len(list_of_files)):
                filename = list_of_files[k]        
                print filename
                out_dur_file = os.path.join(out_dir, filename + '.cmp')
                op1 = open(out_dur_file, 'w');
                flength = flens[k]
                file_data = norm_data[idx:idx+flength]
                
                if perform_CTC_classes:
                    file_data_per_phone = np.reshape(file_data, (len(file_data)/out_dim, out_dim))
                    ph_dur_file = file_data_per_phone[:,5]
                    for x in xrange(10):
                        thres = -1+((x+1)*0.2)
                        ph_dur_file[ph_dur_file<thres] = (x+1)*100
                    ph_dur_file[ph_dur_file<3]  = 1100
                    ph_dur_file[ph_dur_file<30] = 1200
                    ph_dur_file = (ph_dur_file/100)
                    ph_class_file = [int(x-1) for x in ph_dur_file]
                    onehot_ph_class_file = np.zeros((len(ph_class_file), 12))
                    onehot_ph_class_file[np.arange(len(ph_class_file)), ph_class_file] = 1
                    file_data = onehot_ph_class_file
                    #file_data = np.int_(file_data)
                    file_data[file_data == 0] = 0.01
                    file_data[file_data == 1] = 0.99
                    file_data = [' '.join(map(str, x)) for x in file_data]
                
                [op1.write(str(x) + '\n') for x in file_data]    
                idx = idx + flength
                op1.close();
                #break;
                        
        if analysisDur:
            
            rs_norm_data = np.reshape(norm_data, (len(norm_data)/out_dim, out_dim))
            
            print rs_norm_data.shape
            print len(rs_norm_data[rs_norm_data[:,5]<0])
            print len(rs_norm_data[rs_norm_data[:,5]==0])
            print len(rs_norm_data[rs_norm_data[:,5]>0])
            print len(rs_norm_data[rs_norm_data[:,5]>3])
            print len(rs_norm_data[rs_norm_data[:,5]>5])
            print len(rs_norm_data[rs_norm_data[:,5]>8])
            print len(rs_norm_data[rs_norm_data[:,5]>10])
            print len(rs_norm_data[rs_norm_data[:,5]>15])
            
            for i in xrange(11):
                thres = -1+(i*0.2)
                print thres, len(rs_norm_data[rs_norm_data[:,5]<thres])
            
        ph_all_files = []    
        org_dur_all_files = []
        pred_dur_all_files = []
                
        if denormDur:
            gen_dir = os.path.join(work_dir, 'Data/inter-module/'+speaker+'/gen/dur_cmp/')
            stat_fname = feat_dir_path + '.txt'
            stats_file = os.path.join(work_dir, 'Data/inter-module/'+speaker+'/misc/', stat_fname)
            filelist = os.path.join(work_dir, 'Data/fileList/'+speaker+'_test.scp')
            list_of_files = io_funcs.load_file_list(filelist)
            
            max_diff = 0
            for i in range(len(list_of_files)):
                filename = list_of_files[i]
                print filename
                
                org_dur_features = [];
                pred_dur_features = [];
                in_lab_file  = os.path.join(label_align_dir, filename + '.lab')
                [phone, st_arr, ph_arr, mean_f0_arr] = htsclass.read_state_align_label_file(in_lab_file)
                num_of_phones = len(phone) - 2
                
                for j in range(len(phone)):
                    if(phone[j] == '#'):
                        continue;
                    
                    ph_all_files.append(phone[j])
                    for k in range(5):
                        org_dur_features.append(st_arr[j][k]/(np.power(10,4)*5))
                
                org_dur_features = np.array(org_dur_features).reshape(num_of_phones, 5)
                org_dur_all_files = np.concatenate((org_dur_all_files, 
                                                    np.sum(np.array(org_dur_features), axis=1)), axis=0) 
                
                
                #print np.sum(np.array(org_dur_features), axis=1)
                 
                ### load generated output ###
                # gen_data = io_funcs.load_binary_file(gen_file, 1)
                gen_file = os.path.join(gen_dir, filename + '.cmp')
                gen_data = io_funcs.load_float_file(gen_file)
                                
                ##### denormalization of data #####
                print 'denormalizing the data....'
                if(normalization == "MVN"):
                    denorm_data = data_normalization.MVN_denormalize(gen_data, out_dim, stats_file)
                    #gen_data = io_funcs.load_float_file(gen_file)
                    #bkp_denorm_data = data_normalization.MVN_denormalize(gen_data, out_dim, stats_file)
                else:
                    denorm_data = gen_data
                
                denorm_data = np.array(denorm_data)
                #bkp_denorm_data = np.array(bkp_denorm_data)
                
                ### post-processing of state-duration modification ###
                if out_dim==8:
                    for j in xrange(num_of_phones):
                        
                        ### word duration ###
                        st_ph_indx = int(mean_f0_arr[j+1][4]);
                        fn_ph_indx = int(mean_f0_arr[j+1][5]);
                        if(st_ph_indx == 0 or fn_ph_indx == 0):
                            st_ph_indx = j+1
                            fn_ph_indx = j+1
                        
                        st_indx = np.array([0])
                        for x in xrange(4, out_dim+1):
                            st_indx = np.concatenate((st_indx, np.array(range(st_ph_indx, fn_ph_indx+1))*out_dim - x), 0)
                        st_indx  = st_indx[1:]    
                        wrd_indx = np.array(range(st_ph_indx, fn_ph_indx+1))*out_dim - 1
                        ph_indx  = np.array(range(st_ph_indx, fn_ph_indx+1))*out_dim - 3
                        
                        sst_dur = sum(denorm_data[out_dim*j:out_dim*j+5])
                        ph_dur  = denorm_data[out_dim*j+5]
                        wrd_dur = denorm_data[out_dim*j+7]
                        #mean_wrd_dur = (np.mean(denorm_data[wrd_indx]) + np.median(denorm_data[wrd_indx]))/2
                        mean_wrd_dur = np.mean(denorm_data[wrd_indx])
                        sum_ph_wrd_dur = np.sum(denorm_data[ph_indx]) 
                        sum_st_wrd_dur = np.sum(denorm_data[st_indx])
                        
                        mod_factor = (2*mean_wrd_dur)/(mean_wrd_dur+wrd_dur)
                        
                        diff_ph_st = np.abs(round(ph_dur) - round(sst_dur))
                        if(diff_ph_st > max_diff):
                            max_diff = diff_ph_st
                            
                        #print wrd_dur, mean_wrd_dur, sum_st_wrd_dur, sum_ph_wrd_dur, mod_factor
                        for k in xrange(0, 5):  
                            denorm_data[out_dim*j+k] *= mod_factor
                             
                pred_dur_features = [int(round(x)) if x>1 else 1 for x in denorm_data]
                pred_dur_features = np.array(pred_dur_features).reshape(num_of_phones, out_dim)
                            
                #print np.concatenate((org_dur_features.T, pred_dur_features.T), 0).T
                #print np.sum(pred_dur_features[:,0:5], axis=1)  
                pred_dur_all_files = np.concatenate((pred_dur_all_files, 
                                                     np.sum(pred_dur_features[:,0:5], axis=1)), axis=0)
                #pred_dur_all_files = np.concatenate((pred_dur_all_files, 
                #                                     np.sum(pred_dur_features[:,5:6], axis=1)), axis=0)      
            
                if writeDur:
                    if not os.path.exists(out_dur_dir):
                        os.makedirs(out_dur_dir)
                    
                    output_dur_file = os.path.join(out_dur_dir, filename+ '.dur')
                    
                    op1 = open(output_dur_file,'w')
                    for j in range(len(pred_dur_features)):
                        fstr = ' '.join(map(str, pred_dur_features[j][0:5]))
                        op1.write(fstr+'\n')
                    op1.close()
                    
        if modifyLabels:
            if not os.path.exists(out_mod_lab_dir):
                os.makedirs(out_mod_lab_dir)
            filelist = os.path.join(work_dir, 'Data/fileList/'+speaker+'_test.scp')
            list_of_files = io_funcs.load_file_list(filelist)
            modify_phone_labels(label_align_dir, out_dur_dir, out_mod_lab_dir, list_of_files)
        
        if skip_pause:
            ph_all_files = np.array(ph_all_files)
            org_dur_all_files[ph_all_files=='pau'] = 0
            pred_dur_all_files[ph_all_files=='pau'] = 0
            org_dur_all_files = org_dur_all_files[org_dur_all_files>0]
            pred_dur_all_files = pred_dur_all_files[pred_dur_all_files>0]
            
                
        if calcRMSE:
            ### evaluation metrics ###
            rmse_error = eval_metrics.rmse(org_dur_all_files, pred_dur_all_files)
            print 'RMSE: ' + str(rmse_error) 
            print 'CORR: ' + str(eval_metrics.corr(org_dur_all_files, pred_dur_all_files))
