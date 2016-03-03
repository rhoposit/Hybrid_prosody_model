'''
Created on 9 Nov 2015

@author: Srikanth Ronanki
'''
import os
from io_funcs.binary_io import BinaryIOCollection
from io_funcs.read_hts_label_file import readHTSlabelFile
import numpy as np

class linguistic_features():
    
    def __init__(self, vowelListPath):
        ip1 = open(vowelListPath,'r')
        self.vlist = [x.strip() for x in ip1.readlines()]
        ip1.close() 
        pass
    
    def load_word_embeddings(self, word_embed_file):
        self.wrd_embeds = {}
        ip1 = open(word_embed_file, 'r')
        for i in ip1.readlines():
            fstr = i.strip().split()
            word_vec = ' '.join(map(str, fstr[1:]))
            self.wrd_embeds[fstr[0]] = word_vec
        ip1.close()
        
    def extract_base_features(self, feat_dir_path, feat_switch, list_of_files, decomposition_unit, unit_dim):
        ### load Binary module ###
        io_funcs = BinaryIOCollection()
        htsclass = readHTSlabelFile()
        
        ### read file by file ###
        for i in range(len(list_of_files)):    
            filename = list_of_files[i]     
            print filename
            
            binary_label_dir = feat_dir_path['input_binary']
            label_align_dir = feat_dir_path['input_labfile']
            txt_dir = feat_dir_path['input_txt']
            out_feat_dir = feat_dir_path['output_feat']
            
            in_filename = os.path.join(binary_label_dir, filename + '.lab');
            in_lab_file = os.path.join(label_align_dir, filename + '.lab')
            in_txt_file = os.path.join(txt_dir, filename + '.txt')
            out_filename = os.path.join(out_feat_dir, filename + '.lab');
            
            word_embed_list = []
            binary_feat_list = []
            identity_vec_list = []
            dur_feat_list = []
            dur_list = []
            
            ### read text file ###
            if feat_switch['wordEmbed']:
                ip1 = open(in_txt_file, 'r')
                text_Data = ip1.readlines()
                ip1.close()
                
                norm_text = self.format_text(text_Data[0].strip())
                norm_text = norm_text.replace('OUF', 'O U F')
                norm_text = norm_text.replace('Mmm', 'M m m')
                norm_text = norm_text.replace('USA', 'U S A')
                list_of_words = norm_text.split()

            ### read label file ###
            [phone, st_arr, ph_arr, mean_f0_arr] = htsclass.read_state_align_label_file(in_lab_file)
            file_len = len(phone)
            
            ### read binary label file ###
            features = io_funcs.load_binary_file(in_filename, 1)
            
            ### take non-silence region ###
            ph_start = int(ph_arr[0][1] / (np.power(10, 4) * 5));
            ph_end = int(ph_arr[1][file_len-2] / (np.power(10, 4) * 5));
            
            ### extract duration features ###
            frame_feat_list = features.reshape(len(features)/unit_dim['frame'], unit_dim['frame'])
            frame_feat_list = frame_feat_list[ph_start: ph_end, :]
            dur_feat_list   = frame_feat_list[:,-9:]
            
            ### initialise common variables ###
            num_of_frames=0;
            
            ### initialise syllable variables ###
            #frame_indx=0;
            syl_num_of_frames=0
            wc = 0; phinsyl=0;
            syl_identity = self.zeros(300,1)
            syl = ''
            
            j=0;
            while j < file_len: 
                #### ignore silence ####
                if(phone[j] == '#' or phone[j] == 'pau'):
                    j = j + 1
                    continue;            
                
                ### extract boundaries of phone ###
                ph_start = int(ph_arr[0][j] / (np.power(10, 4) * 5));
                ph_end = int(ph_arr[1][j] / (np.power(10, 4) * 5));
                num_of_frames = sum(st_arr[j][:]/(np.power(10,4)*5))
                mid_frame = (ph_start+ph_end)/2
                
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
                
                ### extract binary phone-level features ###
                st_indx = unit_dim['frame']*mid_frame
                mid_frame_feat = features[st_indx:st_indx+592]
                mid_frame_feat = np.reshape(mid_frame_feat, len(mid_frame_feat))
                
                ### word embedding features ###
                if feat_switch['wordEmbed']:            
                    ### word embeddings for syllable ###
                    word = list_of_words[wc]
                    if(word_end and phone[j]!='pau'): 
                        wc += 1    
                    if(phone[j] == 'pau'):
                        word_vec = self.wrd_embeds['*UNKNOWN*']
                    elif word in self.wrd_embeds:
                        word_vec = self.wrd_embeds[word]
                    elif word.lower() in self.wrd_embeds:
                        word_vec = self.wrd_embeds[word.lower()]
                    else:
                        word_vec = self.wrd_embeds['*UNKNOWN*']
                
                ### identity features ###
                if feat_switch['identity']:                
                    ### phone identity features ###
                    ph_identity = mid_frame_feat[99:148]
                    
                    if decomposition_unit == 'syllable':
                        ### syllable identity features 
                        st_indx = phinsyl*50
                        syl_identity[st_indx:st_indx+49] = ph_identity
                        syl = syl + phone[j]
                        ### to make nucleus centre ###
                        #if phone[j] in self.vlist:
                        #    vow_index = phinsyl
                        
                        ### if silence is allowed ###
                        #if phone[j] == '#':
                        #    syl_identity[(phinsyl+1)*50-1] = 1
                        phinsyl += 1
                
                #### select features depending on decomposition unit ###
                
                ### frame-level features ###
                if(decomposition_unit=='frame'):
                    
                    ### duration features for phone ###
                    dur_list.append(num_of_frames)
                    
                    ### frame level binary features ###
                    if feat_switch['binary'] and j+2==file_len:
                        ### load normalisation statistics ###
                        label_norm_float_file = os.path.join(binary_label_dir, '../label_norm_float_HTS.dat');
                        fid = open(label_norm_float_file, 'r')
                        arr12 = [float(x.strip()) for x in fid.readlines()]
                        fid.close()
                        min_vector = np.array(arr12[0:len(arr12)/2])
                        max_vector = np.array(arr12[len(arr12)/2:len(arr12)])
                        max_range_vector = max_vector - min_vector
                        max_range_vector[max_range_vector==0] = 1
                        
                        ### normalise features ###
                        nrows = len(frame_feat_list)
                        for x in xrange(nrows):
                            norm_frame_feat = (frame_feat_list[x,:] - min_vector) / max_range_vector*0.98 + 0.01
                            norm_frame_vec = ' '.join(map(str, norm_frame_feat[:]))
                            binary_feat_list.append(norm_frame_vec)
                    
                    ### embedding features ###
                    if feat_switch['wordEmbed']:
                        for x in xrange(num_of_frames):
                            word_embed_list.append(word_vec)
                        
                ### phone-level features ###
                if(decomposition_unit=='phone'):
                    
                    ### duration features for phone ###
                    dur_list.append(num_of_frames)
                    
                    ### phone level binary features ###
                    if feat_switch['binary']:
                        #ph_feat = np.concatenate((mid_frame_feat[0:99], mid_frame_feat[348:]), axis=0)
                        norm_ph_feat = [0.99 if x==1 else 0.01 for x in mid_frame_feat]
                        norm_ph_vec = ' '.join(map(str, norm_ph_feat[:]))
                        binary_feat_list.append(norm_ph_vec)
                    
                    ### embedding features ###
                    if feat_switch['wordEmbed']:
                        word_embed_list.append(word_vec)
                    
                    ### phone-identity features ###
                    if feat_switch['identity']:
                        extra_ph = 1 if phone[j] == 'o~' else 0
                        ph_identity = np.append(ph_identity, extra_ph)
                        #norm_ph_identity = [0.99 if x==1 else 0.01 for x in ph_identity]
                        norm_ph_identity = [int(x) for x in ph_identity]
                        norm_ph_identity_vec = ' '.join(map(str, norm_ph_identity[:]))
                        identity_vec_list.append(norm_ph_identity_vec)
                
                
                ### syllable level features ###
                if(decomposition_unit=='syllable' and syl_end):
                    #print syl
                    
                    ### duration features for syllable ###
                    dur_list.append(syl_num_of_frames)
                    
                    ### syllable and above level binary features ###
                    if feat_switch['binary']:
                        syl_feat = []
                        for x in range(len(mid_frame_feat)):
                            if(x < 348 or (x >= 405 and x < 421)):
                                continue;
                            syl_feat.append(mid_frame_feat[x])
                        norm_syl_feat = [0.99 if x==1 else 0.01 for x in syl_feat]
                        norm_syl_vec = ' '.join(map(str, norm_syl_feat[:]))
                        binary_feat_list.append(norm_syl_vec)
                    
                    if feat_switch['wordEmbed']:
                        word_embed_list.append(word_vec)
                    
                    ### syllable-identity features ###
                    if feat_switch['identity']:
                        ### to make nucleus centre ###
                        #if(vow_index<=1):
                        #    syl_identity = np.roll(syl_identity, 50*(vow_index+1)) 
                        norm_syl_identity = [0.99 if x==1 else 0.01 for x in syl_identity]
                        norm_syl_identity_vec = ' '.join(map(str, norm_syl_identity[:]))
                        identity_vec_list.append(norm_syl_identity_vec)
                        
                    ### reset syllable information ###
                    phinsyl = 0; syl=''
                    syl_num_of_frames = 0 
                    syl_identity = self.zeros(300, 1)    
                
                j+=1                   
            
            ### default vectors to use ###
            if feat_switch['identity'] and decomposition_unit=='syllable': 
                syl_identity = self.zeros(300, 1)
                norm_syl_identity = [0.99 if x==1 else 0.01 for x in syl_identity]
                norm_syl_identity_vec = ' '.join(map(str, norm_syl_identity[:]))
            if feat_switch['wordEmbed']:
                word_vec = self.wrd_embeds['*UNKNOWN*']
                
            
            ### writing features to output file ###
            op1 = open(out_filename, 'w')
            num_of_vectors = max(len(binary_feat_list), len(identity_vec_list), len(word_embed_list))
            for x in range(num_of_vectors):
                ### initialise feat vector ###
                feat_vec = ''
                
                ### binary features ###
                if feat_switch['binary']:
                    feat_vec = feat_vec + binary_feat_list[x]+' '
                    
                ### word embeddings ###
                if feat_switch['wordEmbed']:
                    if feat_switch['wordEmbed']>=3:
                        if(x-1<0):
                            feat_vec = feat_vec + word_vec+' '
                        else:
                            feat_vec = feat_vec + word_embed_list[x-1]+' '
                    feat_vec = feat_vec + word_embed_list[x]+' '
                    if feat_switch['wordEmbed']>=3:
                        if(x+1>=len(binary_feat_list)):
                            feat_vec = feat_vec + word_vec+' '
                        else:
                            feat_vec = feat_vec + word_embed_list[x+1]+' '
                
                ### identity features ###
                if feat_switch['identity']:
                    if feat_switch['identity']>=5:
                        if(x-2<0):
                            feat_vec = feat_vec + norm_syl_identity_vec+' '
                        else:
                            feat_vec = feat_vec + identity_vec_list[x-2]+' '
                    if feat_switch['identity']>=3:
                        if(x-1<0):
                            feat_vec = feat_vec + norm_syl_identity_vec+' '
                        else:
                            feat_vec = feat_vec + identity_vec_list[x-1]+' '
                    feat_vec = feat_vec + identity_vec_list[x]+' '
                    if feat_switch['identity']>=3:
                        if(x+1>=len(binary_feat_list)):
                            feat_vec = feat_vec + norm_syl_identity_vec+' '
                        else:
                            feat_vec = feat_vec + identity_vec_list[x+1]+' '
                    if feat_switch['identity']>=5:
                        if(x+2>=len(binary_feat_list)):
                            feat_vec = feat_vec + norm_syl_identity_vec+' '
                        else:
                            feat_vec = feat_vec + identity_vec_list[x+2]+' '
                op1.write(feat_vec+'\n')
                #for z in range(dur_list[x]):
                #    op1.write(feat_vec + ' '.join(map(str, dur_feat_list[frame_indx+z,:]))+'\n')
                #frame_indx+=dur_list[x]
            op1.close()    
            #break;
        
    def zeros(self, m, n):
        if(n == 1):
            arr = np.ndarray((m,), float)
        else:
            arr = np.ndarray((m, n), float)
        arr.fill(0)
        return arr
    
    def format_text(self, input_str):
        ### upper case to lower case ###
        input_str = input_str.replace("'","2211")
        
        fp = open('temp','w')
        fp.write(input_str+'\n')
        fp.close()
        
        ### remove punctuation marks ###
        os.system('sed -i \'s/\([[:punct:]]\)/ /g\' temp')

        ### remove trailing spaces ###
        os.system('sed -i -e \'s/^[ \t]*//;s/[ \t]*$//\' temp')

        ### remove multiple spaces ###
        os.system('sed -i \'s/  */ /g\' temp')
        
        fp = open('temp','r')
        for i in fp.readlines():
            output_str = i.strip()
        fp.close()

        output_str = output_str.replace("2211","'")
        return output_str
    
if __name__ == "__main__":
    
    speaker = 'blz16'
    
    work_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Hybrid_prosody_model'
    dnn_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/LSTM_work/dnn_tts_'+speaker+'ehmm'
        
    feat_switch = {}
    feat_switch['binary']     = 1
    feat_switch['wordEmbed']  = 1
    feat_switch['identity']   = 3
    feat_switch['bottleneck'] = 0
    
    unit_dim = {}
    unit_dim['frame']    = 601
    unit_dim['phone']    = 592
    unit_dim['syllable'] = 228
    unit_dim['word']     = 92
        
    decomposition_unit = 'syllable'
    
    word_embed_size = 50*feat_switch['wordEmbed']
    
    identity_size=0
    if decomposition_unit == 'phone':
        identity_size = 50*feat_switch['identity']
    elif decomposition_unit == 'syllable':
        identity_size = 300*feat_switch['identity']
    
    baseline_size = unit_dim[decomposition_unit]*feat_switch['binary']
    
    in_dim  = 601
    out_dim = word_embed_size+identity_size+baseline_size
 
    binary_label_dir = os.path.join(dnn_dir, 'lstm_rnn/data/binary_label_' + str(in_dim))
    label_align_dir = os.path.join(work_dir, 'Data/inter-module/'+speaker+'/label_state_align')  
    text_dir = os.path.join(work_dir, 'Data/database/'+speaker+'/txt/')
    
    out_dir = decomposition_unit+'_baseline_'+str(baseline_size)+'_wordembed_'+str(word_embed_size)+'_identity_'+str(identity_size)
    out_feat_dir = os.path.join(work_dir, 'Data/inter-module/'+speaker+'/label_features/' + str(out_dir) + '/binary_label_' + str(out_dim))
    if not os.path.exists(out_feat_dir):
        os.makedirs(out_feat_dir)
        
    feat_dir_path = {}
    feat_dir_path['input_binary']  = binary_label_dir
    feat_dir_path['input_labfile'] = label_align_dir
    feat_dir_path['input_txt']     = text_dir
    feat_dir_path['output_feat']   = out_feat_dir

    DFP = 1
    if DFP:
        extract_base_feats = True;
    
        filelist = os.path.join(work_dir, 'Data/fileList/'+speaker+'.scp')
        word_embed_file = os.path.join(work_dir, 'Data/word_embeddings/turian-embeddings-50.txt')
        vowelListPath = os.path.join(work_dir, 'Data/phoneset/vowels.txt')
        
        ip_feats = linguistic_features(vowelListPath)
        io_funcs = BinaryIOCollection()
        
        if extract_base_feats: 
            list_of_files = io_funcs.load_file_list(filelist)
            
            if feat_switch['wordEmbed']:
                print 'loading word embeddings...'
                ip_feats.load_word_embeddings(word_embed_file)
                
            print out_feat_dir
            print 'extracting features...'
            ip_feats.extract_base_features(feat_dir_path, feat_switch, list_of_files, decomposition_unit, unit_dim)
        
