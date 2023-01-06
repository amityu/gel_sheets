
class Movie:


    '''
    holds several time points
    '''


class Pcluster:

    '''

    '''

    #  ------------ no multiple constructir
    '''def __init__(self, rr,cc, image):

        

        @param rr:  pixels list row indices
        @type rr:
        @param cc: pixels list column indices, same length as rr
        @type cc:
        
        self.rr = rr
        self.cc = cc
        self.size = len(rr)    # number of pixels in cluster
        self.image =  image
        self.max_intensity = np.nanmax(self.image[self.rr,self.cc])
        self.mean_intensity= np.nanmean(self.image[self.rr, self.cc])
    '''

    def __init__(self, indices_array, image):
        '''

        @param indices_array: m,2 numpy array of indices
        '''
        self.rr = indices_array[:,0]
        self.cc = indices_array[:,1]

        self.size = len(indices_array)  # number of pixels in cluster
        self.image = image
        self.max_intensity = np.nanmax(self.image[self.rr, self.cc])
        self.mean_intensity= np.nanmean(self.image[self.rr, self.cc])
        self.region = 0
        self.out_boundary = np.nan
    def set_in_boundary(self):
        image = np.zeros(self.image.shape, dtype = bool)
        image[self.rr,self.cc] = True
        self.in_boundary = np.nonzero(find_boundaries(image, mode = 'inner'))

    def set_out_boundary(self):
        image = np.zeros(self.image.shape, dtype = bool)
        image[self.rr,self.cc] = True
        self.out_boundary = np.nonzero(find_boundaries(image, mode = 'outer'))

class Pframe:
    '''
    self.distribution : 2d numpy array with relative cluster sizes (relative to gel area ) , and number of clusters of this size
    '''

    def __init__(self, image,sigma = np.nan):
        '''

        @param image: 2d numpy array
        @param sigma: if sigma isnan uses image nanmean instead
        '''
        if np.isnan(sigma): sigma = np.nanmean(image)
        self.image = np.zeros(image.shape)
        np.copyto(self.image,image)
        self.binary = image > sigma
        self.gel_area = np.sum(~np.isnan(image))   # area of the gel at this time frame
        self.gel_mask = ~np.isnan(image)
        self.clusters = []
    def set_clusters(self, binary =np.nan):

        '''
        Adds clusters to self.clusters
        returns the list of clusters added this run
        '''
        current_clusters = []
        if binary is np.nan:
            tmp_binary = self.binary.copy()
        else:
            tmp_binary = binary.copy()
        occupied = np.transpose(np.nonzero(tmp_binary)).tolist()
        foot = np.array([[0,1,0], [1,1,1],[0,1,0]])
        #foot = np.ones((3,3))
        binary_sum = tmp_binary.sum()
        total_cluster = 0
        while len(occupied)>0:
            cluster_image = flood(tmp_binary, seed_point=tuple(occupied[0]),selem=foot)
            if  np.any(cluster_image):
                new_cluster = np.transpose(np.nonzero(cluster_image))
                current_clusters.append(Pcluster(new_cluster, self.image))
                total_cluster += Pcluster(new_cluster, self.image).size
                for l in range(len(new_cluster)):
                    occupied.remove(new_cluster[l].tolist())
                #print(new_cluster)

        self.clusters += current_clusters
        return current_clusters
        # ---------------------  distribution ------------------------------------------


    def set_distributation(self):
        bins = 50 # number of bins in histogram
        self.cluster_sizes = np.zeros(len(self.clusters))
        for k in range(len(self.clusters)):
            self.cluster_sizes[k] = self.clusters[k].size

        self.cluster_sizes /= (~np.isnan(self.image)).sum()

        if len(self.clusters)==0:
            self.distribution = np.zeros(bins)
        else:
            hist, bin_edges= np.histogram(self.cluster_sizes,bins = bins,range = (0,1),density=True)
            #self.distribution = np.concatenate((np.arange(1,bins+1)/50,hist.Tns,np.array([hist[hist!=0]]).T),axis=1)
            self.distribution = np.concatenate((((np.arange(1,bins+1)-0.5)/bins).reshape((-1,1)), hist.reshape((-1,1))), axis=1)

    def set_binary_range(self, min,max):
        '''
        sets the binary to be in the range of the normalized image

        @param min: min range as a number of standard deviations from mean
        @param max: max range as a number of standard deviations from mean
        @return: 0 if succesful
        '''
        normalized_image = ((self.image - np.nanmean(self.image))/np.nanstd(self.image)).copy()
        self.binary = np.bitwise_and(normalized_image > min,normalized_image<max)
        return 0




def binary_cluster_size_histogram(I, gel_parameter):
    '''

    @param I: gel
    @param gel_parameter:
    @return: saves np array of cluster size histogram and figure at My_Graph_path/cluster_histogram/
    '''
    bins = 50

    number_of_frames = 100
    x = np.arange(-0.5, number_of_frames + 0.5, 1)
    y = np.arange(-0.5, bins + 0.5, 1)

    cmap = plt.get_cmap('winter')
    Z = np.zeros((number_of_frames, bins))
    k = 0
    for t in tqdm((np.linspace(0, gel_parameter['end'], number_of_frames, endpoint=False)).astype(int)):
        print(gel_parameter['name'], 't=%d' % (t))

        my_Pframe = Pframe(I[t])
        my_Pframe.set_clusters()
        my_Pframe.set_distributation()

        cluster_pixel_count = my_Pframe.distribution[:, 1] * my_Pframe.distribution[:, 0]
        normalized_pixel_count = cluster_pixel_count / np.nansum(cluster_pixel_count)
        Z[k] = normalized_pixel_count
        Z[Z == 0] = np.nan
        k += 1

    fig, ax = plt.subplots()
    c = ax.pcolormesh(y, x, Z, cmap=cmap)
    plt.colorbar(c, ax=ax)
    plt.xlabel('cluster size * number of clusters')
    plt.ylabel('time frame')
    plt.title(gel_parameter['name'] + 'cluster histogram')
    plt.savefig(MY_GRAPH_PATH + 'cluster_histogram/' + gel_parameter['name'] + 'cluster_hist.png')
    plt.show()
    np.save(MY_GRAPH_PATH + 'cluster_histogram/' + gel_parameter['name'] + 'cluster_size_total_norm.npy', Z)
    graph_parameter = {'number_of_frames': number_of_frames, 'bins': bins}
    with open(MY_GRAPH_PATH + 'cluster_histogram/' + gel_parameter['name'] + 'cluster_size_total_norm_para.json',
              'w') as outfile:
        json.dump(graph_parameter, outfile)


def multi_cluster_size_histogram(I, gel_parameter):
    '''

    @param I: gel
    @param gel_parameter:
    @return: np array of cluster size histogram and figure at My_Graph_path/cluster_histogram/
            +values dictionary
    '''

    #??????????? change function to read from directory
    bins = 50
    Q = 255

    number_of_frames = 100

    x = np.arange(-0.5, number_of_frames + 0.5, 1)
    y = np.arange(-0.5, bins + 0.5, 1)

    cmap = plt.get_cmap('winter')
    Z = np.zeros((number_of_frames, bins))
    k = 0
    #for t in tqdm((np.linspace(0, gel_parameter['end'], number_of_frames, endpoint=False)).astype(int)):
    for t in tqdm(range(356, gel_parameter['end'],100)):

        print(gel_parameter['name'], 't=%d' % (t))

        my_Pframe = Pframe(I[t])
        #my_srm = stat_region_merge.Srm(I[t], Q= Q)
        #region_map = my_srm.cluster()
        region_map =  np.load(MY_GRAPH_PATH + 'stat_clustering/'+gel_parameter['name'] +'cluster%05d_sorted_01.npy' % t)

        #for region in range(np.max(region_map)):
        for region in np.unique(region_map[~np.isnan(region_map)]):
            my_Pframe.binary = (region_map==0)
            my_Pframe.set_clusters(binary=(region_map == region))

            my_Pframe.set_distributation()

            cluster_pixel_count = my_Pframe.distribution[:, 1] * my_Pframe.distribution[:, 0]
            normalized_pixel_count = cluster_pixel_count / np.nansum(cluster_pixel_count)
            Z[k] = normalized_pixel_count
            Z[Z == 0] = np.nan
            fig, ax = plt.subplots()
            c = ax.pcolormesh(y, x, Z, cmap=cmap)
            plt.colorbar(c, ax=ax)
            plt.xlabel('cluster size * number of clusters')
            plt.ylabel('time frame')
            plt.title(gel_parameter['name'] + 'cluster histogram')
            plt.savefig(MY_GRAPH_PATH + 'cluster_histogram/' + gel_parameter['name'] + 'cluster_hist_multy.png')
            plt.show()
            np.save(MY_GRAPH_PATH + 'cluster_histogram/' + gel_parameter['name'] + 'cluster_size_total_norm_multi.npy',
                    Z)
            graph_parameter = {'number_of_frames': number_of_frames, 'bins': bins, 'region': region}

            with open(MY_GRAPH_PATH + 'cluster_histogram/' + gel_parameter[
                'name'] + 'cluster_size_total_norm_multi_para.json',
                      'w') as outfile:
                json.dump(graph_parameter, outfile)
        k += 1


        #return Z, graph_parameter


def show_cluster_size_histogram(gel_parameter):
    '''
    shows the kymograph of cluster size
    @param gel_parameter:
    @return:
    '''
    f = open(MY_GRAPH_PATH + 'cluster_histogram/' + gel_parameter['name'] + 'cluster_size_total_norm_para.json')
    graph_parameter = json.load(f)
    f.close()
    bins = graph_parameter['bins']
    number_of_frames = graph_parameter['number_of_frames']
    cmap = plt.get_cmap('plasma')
    x = np.arange(-0.5, number_of_frames + 0.5, 1)
    y = np.arange(-0.5, bins + 0.5, 1)
    Z = np.load(MY_GRAPH_PATH + 'cluster_histogram/' + gel_parameter['name'] + 'cluster_size_total_norm.npy')
    Z[Z == 0] = np.nan

    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, y, Z.T, cmap=cmap, vmin=0, vmax=1)
    clb = plt.colorbar(c, ax=ax, )
    clb.set_label('Total cluster size fraction of gel size')
    plt.ylabel('Each cluster size fraction %')
    plt.xlabel('Time frame')
    plt.xticks(range(0, number_of_frames, 30),
               (np.arange(0, number_of_frames, 30) * int(gel_parameter['end'] / number_of_frames)).astype(int))
    plt.yticks(range(0, bins, 10), range(0, 100, int(100 / bins * 10)))
    plt.title('cluster size distribution \n' + gel_parameter['name'])
    plt.savefig(MY_GRAPH_PATH + 'cluster_size_' + gel_parameter['name'] + '.png')
    plt.show()

def n_fold_clustering(I,gel_parameter, n_folds = 8, step = 100):

    for t in tqdm(range(len(I)-1, 100, -step)):
        frame_min = np.nanmin(I[t])
        frame_max = np.nanmax(I[t])
        frame_range = frame_max - frame_min
        region_map = np.zeros(I[t].shape)
        region_map[region_map==0] = np.nan
        for fold in range(n_folds):
            frame = Pframe(I[t])
            fold_min = frame_min + (fold/n_folds)*frame_range
            fold_max = frame_min + (fold+1 / n_folds) * frame_range

            binary = np.bitwise_and((I[t]>= fold_min), I[t] <fold_max)
            frame.set_clusters(binary)
            for cluster in  frame.clusters:
                region_map[cluster.rr, cluster.cc] = fold
        np.save(MY_GRAPH_PATH + 'eight_fold_clustering/' + gel_parameter['name'] + '_t_%d.npy'%t, region_map)
        new_region_map = np.zeros(region_map.shape)

        new_region_map = region_map /n_folds
        cmap = copy.copy(plt.get_cmap('seismic'))
        cmap.set_bad('green')
        plt.imshow(new_region_map, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar()
        plt.title(gel_parameter['name'] + ' Time_frame = %d, %d folds'%(t,n_folds) )
        plt.savefig(MY_GRAPH_PATH + 'eight_fold_clustering/' + gel_parameter['name'] + '_t_%d_%d_folds.png'%(t, n_folds))
        plt.show()


def n_fold_cluster_distribution(I,gel_parameter, n_folds = 8, step = 100):
    t_list = []
    size_list = []
    fold_list = []
    for t in tqdm(range(len(I)-1, 100, -step)):
        frame_min = np.nanmin(I[t])
        frame_max = np.nanmax(I[t])
        frame_range = frame_max - frame_min
        region_map = np.zeros(I[t].shape)
        region_map[region_map==0] = np.nan
        for fold in range(n_folds):
            frame = Pframe(I[t])
            fold_min = frame_min + (fold/n_folds)*frame_range
            fold_max = frame_min + (fold+1 / n_folds) * frame_range

            binary = np.bitwise_and((I[t]>= fold_min), I[t] <fold_max)
            frame.set_clusters()

            for cluster in  frame.clusters:
                t_list.append(t)
                size_list.append( cluster.size)
                fold_list.append((fold+0.5-n_folds/2)*2)

    return pd.DataFrame(list(zip(t_list,size_list,fold_list)),columns = ['t','size','fold'])
class Region_map(Pframe):

    def __init__(self, image, region_map_path):
        super().__init__(image)
        self.region_map= np.load(region_map_path)

    def __init__(self, image, region_map):
        super().__init__(image)
        self.region_map= region_map

    def set_clusters(self):
        '''
        set all clusters of all regions of region map
        enter region to cluster according to region map
        '''
        self.clusters = []
        for region in range(int(np.nanmax(self.region_map))+1):
            self.binary = self.region_map == region

            new_clusters = super().set_clusters(self.binary)
            for cluster in new_clusters:
                cluster.region = region

    def get_cluster_by_region(self, region):
        '''
        @param region: the region number you want all cluster of
        return a list of cluster of region - region

        '''
        clusters_of_region = []
        for cluster in self.clusters:
            if cluster.region == region:
                clusters_of_region.append(cluster)
        return clusters_of_region





