import os
import numpy as np
from io_funcs.binary_io import BinaryIOCollection
from io_funcs.read_hts_label_file import readHTSlabelFile


if __name__ == "__main__":
    ###    This is main function   ###    
    ### load all modules ###
    htsclass = readHTSlabelFile()
    io_funcs = BinaryIOCollection()
    
    ### set speaker ###
    speaker = 'blz16'
    
    ### model parameters ###
    in_unit = 'syllable'
    out_unit = 'frame' 
    num_of_clusters = 6
    out_feat_dim = num_of_clusters
    
    ### Relative work path ###
    # work_dir = os.path.join(os.getcwd(), "../../")
    
    ### Absolute work path ###
    work_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Hybrid_prosody_model/'
    if out_feat_dim > num_of_clusters:
        in_feat_dir_path = in_unit + '_template_' + str(num_of_clusters) + '_stat_' + str(out_feat_dim-num_of_clusters)
        out_feat_dir_path = out_unit + '_template_' + str(num_of_clusters) + '_stat_' + str(out_feat_dim-num_of_clusters)
    else:
        in_feat_dir_path = in_unit + '_template_' + str(num_of_clusters)
        out_feat_dir_path = out_unit + '_template_' + str(num_of_clusters)
    
    lab_dir = os.path.join(work_dir, 'Data/inter-module/'+speaker+'/label_state_align/')
    
    
    ### Directory of files processing ###
    DFP = True;
    if DFP:
        templatefeats = True;
        
        if templatefeats:
            
            filelist = os.path.join(work_dir, 'Data/fileList/'+speaker+'.scp')
            list_arr = io_funcs.load_file_list(filelist)
            
            in_template_dir = os.path.join(work_dir, 'Data/inter-module/'+speaker+'/template_features/' + in_feat_dir_path + '/')
            out_template_dir = os.path.join(work_dir, 'Data/inter-module/'+speaker+'/template_features/' + out_feat_dir_path + '/')
            
            if not os.path.exists(out_template_dir):
                os.makedirs(out_template_dir)
            
            removesilence=False
            for k in range(1):
                filename = list_arr[k]
                print filename
                in_file = os.path.join(in_template_dir, filename + '.cmp')
                out_file = os.path.join(out_template_dir, filename + '.cmp')
                
                in_lab_file = os.path.join(lab_dir, filename + '.lab') 
                
                ### read label file ###
                [phone, st_arr, ph_arr, mean_f0_arr] = htsclass.read_state_align_label_file(in_lab_file)
                
                ### read template file ###
                ip1 = open(in_file, 'r')
                template_feat = [float(x.strip()) for x in ip1.readlines()]
                ip1.close()
                
                nrows = len(template_feat)/out_feat_dim
                template_feat = np.array(template_feat).reshape(nrows, out_feat_dim)
                
                ### initialisations ###
                syl_num_of_frames = 0
                wrd_num_of_frames = 0
                syl=''; sylcnt=0
                file_len = len(phone)
                
                ### writing features to output file ###
                op1 = open(out_file, 'w')
                
                j=0;
                while j < file_len: 
                    
                    ### extract boundaries of phone ###
                    ph_start = int(ph_arr[0][j] / (np.power(10, 4) * 5));
                    ph_end = int(ph_arr[1][j] / (np.power(10, 4) * 5));
                    num_of_frames = sum(st_arr[j][:]/(np.power(10,4)*5))
                    
                    #### ignore silence ####
                    if(phone[j] == '#'):
                        j = j + 1
                        if removesilence:
                            continue;            
                        elif(out_unit=='frame'):
                            sil_temp = np.zeros(out_feat_dim)
                            if j == file_len:
                                num_of_frames+=5
                            for z in xrange(num_of_frames):
                                op1.write(' '.join(map(str, sil_temp))+'\n')
                            continue
                    
                    syl = syl + phone[j]
                    
                    ### syllable ending information ###
                    syl_end = 0        
                    if(mean_f0_arr[j + 1][3] - mean_f0_arr[j][3] != 0):
                        syl_end = 1
                    
                    ### word ending information ###
                    word_end = 0        
                    if(mean_f0_arr[j + 1][5] - mean_f0_arr[j][5] != 0):
                        word_end = 1
                    
                    
                    ### syllable duration ###
                    syl_num_of_frames += num_of_frames
                    wrd_num_of_frames += num_of_frames
                    
                    
                    if(in_unit=='syllable' and syl_end):
                        #print sylcnt, syl, syl_num_of_frames
                        
                        if(out_unit=='frame'):
                            for z in xrange(syl_num_of_frames):
                                op1.write(' '.join(map(str, template_feat[sylcnt,:]))+'\n')
                
                        sylcnt+=1
                        syl_num_of_frames=0
                        syl = ''
                    
                    j+=1
                
                op1.close()
                
            
                