from matplotlib import colors, cm
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sb
import scrublet as scr
import scipy
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.colors as cccp
from matplotlib.cm import ScalarMappable
from anndata import AnnData
import requests
import json

phase_cols=['#d73027','#1a9641','#fff404']

def scale_data_5_75(data):
    mind = np.min(data)
    maxd = np.max(data)
    if maxd == mind:
        maxd=maxd+1
        mind=mind-1   
    drange = maxd - mind
    return ((((data - mind)/drange*0.70)+0.05)*100)

def plot_enrich(data, n_terms=20, colorpalette = 'cool', main_title='GO Process', save=False):
    if not isinstance(data, pd.DataFrame):
        raise ValueError('Please input a Pandas Dataframe output by gprofiler.')  
    if not np.all([term in data.columns for term in ['p_value', 'name', 'intersection_size']]):
        raise TypeError('The data frame {} does not contain enrichment results from gprofiler.'.format(data))
    data_to_plot = data.iloc[:n_terms,:].copy()
    data_to_plot['go.id'] = data_to_plot.index
    min_pval = data_to_plot['p_value'].min()
    max_pval = data_to_plot['p_value'].max()
    data_to_plot['scaled.overlap'] = scale_data_5_75(data_to_plot['intersection_size'])
    norm = colors.LogNorm(min_pval, max_pval)
    sm = plt.cm.ScalarMappable(cmap=colorpalette, norm=norm)
    sm.set_array([])
    rcParams.update({'font.size': 14, 'font.weight': 'normal'})
    sb.set(style="whitegrid")
    path = plt.scatter(x='recall', y="name", c='p_value', cmap=colorpalette, 
                       norm=colors.LogNorm(min_pval, max_pval), 
                       data=data_to_plot, linewidth=1, edgecolor="grey", 
                       s=[(i+10)**1.5 for i in data_to_plot['scaled.overlap']])
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_title(main_title, fontsize=14, fontweight='normal')
    ax.set_ylabel('')
    ax.set_xlabel('Gene ratio', fontsize=14, fontweight='normal')
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    min_tick = np.floor(np.log10(min_pval)).astype(int)
    max_tick = np.ceil(np.log10(max_pval)).astype(int)
    tick_step = np.ceil((max_tick - min_tick)/6).astype(int)
    if tick_step == 0:
        tick_step = 1
        min_tick = max_tick-1 
    ticks_vals = [10**i for i in range(max_tick, min_tick-1, -tick_step)]
    ticks_labs = ['$10^{'+str(i)+'}$' for i in range(max_tick, min_tick-1, -tick_step)]
    fig = plt.gcf()
    cbaxes = fig.add_axes([0.8, 0.15, 0.03, 0.4])
    cbar = ax.figure.colorbar(sm, ticks=ticks_vals, shrink=0.5, anchor=(0,0.1), cax=cbaxes)
    cbar.ax.set_yticklabels(ticks_labs)
    cbar.set_label("Adjusted p-value", fontsize=14, fontweight='normal')
    min_olap = data_to_plot['intersection_size'].min()
    max_olap = data_to_plot['intersection_size'].max()
    olap_range = max_olap - min_olap
    size_leg_vals = [np.round(i/5)*5 for i in 
                          [min_olap, min_olap+(20/70)*olap_range, min_olap+(45/70)*olap_range, max_olap]]
    size_leg_scaled_vals = scale_data_5_75(size_leg_vals)
    l1 = plt.scatter([],[], s=(size_leg_scaled_vals[0]+10)**1.5, edgecolors='none', color='black')
    l2 = plt.scatter([],[], s=(size_leg_scaled_vals[1]+10)**1.5, edgecolors='none', color='black')
    l3 = plt.scatter([],[], s=(size_leg_scaled_vals[2]+10)**1.5, edgecolors='none', color='black')
    l4 = plt.scatter([],[], s=(size_leg_scaled_vals[3]+10)**1.5, edgecolors='none', color='black')
    labels = [str(int(i)) for i in size_leg_vals]
    leg = plt.legend([l1, l2, l3, l4], labels, ncol=1, frameon=False, fontsize=12,
                     handlelength=1, loc = 'center left', borderpad = 1, labelspacing = 1.4,
                     handletextpad=2, title='Gene overlap', scatterpoints = 1,  bbox_to_anchor=(-1, 1.5), 
                     facecolor='black')
    if save:
        plt.savefig(save, dpi=300, format='pdf')
    plt.show()
    sc.set_figure_params(scanpy=True, dpi=75, dpi_save=150, frameon=True, vector_friendly=True, fontsize=14, figsize=None, color_map=None, format='pdf', facecolor=None, transparent=False, ipython_format='png2x')

def string_enric(my_genes_in, org, prefix, bgd=False):
    Goterm_process_vec = []
    Goterm_function_vec = []
    string_api_url = "https://string-db.org/api"
    output_format = "json"
    method = "enrichment"
    if org not in ['human','mouse']:
        raise ValueError('The organism must be human or mouse')
        return
    if org == 'human':
        sp_id=9606
    else:
        sp_id=10090
    org_dict={'human':'hsapiens',
             'mouse':'mmusculus'}
    request_url = "/".join([string_api_url, output_format, method])
    params = {
        "identifiers" : "%0d".join(my_genes_in),
        "species" : sp_id,
        "caller_identity" : "www.awesome_app.org"
    }
    response = requests.post(request_url, data=params)
    data = json.loads(response.text)
    for row in data:
        term = row["term"]
        fdr = float(row["fdr"])
        description = row["description"]
        category = row["category"]
        nog = row["number_of_genes"]
        bgr = row['number_of_genes_in_background']
        if category == "RCTM" and fdr < 0.01:   
            Goterm_function_vec.append([term, description, nog, bgr, fdr])
        if category == "Process" and fdr < 0.01:
            Goterm_process_vec.append([term, description, nog, bgr, fdr])    
    gprof = pd.DataFrame.from_records(Goterm_process_vec, columns=['go.id','name', 'intersection_size', 'background', 'p_value'], index = 'go.id').sort_values(by=['p_value'])
    if not gprof.empty:
        gprof['recall'] = gprof['intersection_size'].div(gprof['background'])
        plt.figure(figsize=(4, 11))
        plot_enrich(gprof, colorpalette='viridis', main_title='GO process ' + prefix, save=os.path.join(outpath,'goProcess_'  + prefix))
    gproffunc = pd.DataFrame.from_records(Goterm_function_vec, columns=['go.id','name', 'intersection_size', 'background', 'p_value'], index = 'go.id').sort_values(by=['p_value'])
    if not gproffunc.empty:
        gproffunc['recall'] = gproffunc['intersection_size'].div(gproffunc['background'])
        plt.figure(figsize=(4, 11))  
        plot_enrich(gproffunc, colorpalette='viridis', main_title='Reactome ' + prefix, save=os.path.join(outpath,'Reactome_' + prefix))
    org_gp = org_dict[org]
    paneth_enrichment = gp.profile(organism=org_gp, sources=['GO:BP'], user_threshold=0.01,
                           significance_threshold_method='fdr', 
                           background=bgd, 
                           query=my_genes_in)
    if not paneth_enrichment.empty:
        plt.figure(figsize=(4, 11))
        plot_enrich(paneth_enrichment, colorpalette='plasma', main_title='gProfile ' + prefix, save=os.path.join(outpath,'gProfile_' + prefix))
    return

def replaceNth(s, source, target, n): 
    if bool(re.search(r'[0-9]+.[0-9].', s)): 
        inds = [i for i in range(len(s) - len(source)+1) if s[i:i+len(source)]==source] 
        if len(inds) < n: 
            return 
        s = list(s) 
        s[inds[n-1]:inds[n-1]+len(source)] = target 
        return ''.join(s) 
    else: 
        return s

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
    
def distro_barplot(data_in, xx, yy, xlabel= '', ylabel='', save=False, reorder=None, legend=True, axo=None, rotation = 0, ypalette=sc.pl.palettes.vega_20, xpalette=sc.pl.palettes.vega_20):
    if type(data_in) == AnnData:
        SexRatio_inCluster = pd.DataFrame(list(map(lambda x:data_in[data_in.obs[xx]==x].obs[yy].value_counts(),data_in.obs[xx].cat.categories.to_list())))
        SexRatio_inCluster = SexRatio_inCluster.div(SexRatio_inCluster.sum(axis=1), axis=0).fillna(0)[data_in.obs[yy].cat.categories]
    else:
        SexRatio_inCluster = pd.DataFrame(list(map(lambda x:data_in[data_in[xx]==x][yy].value_counts(),data_in[xx].cat.categories.to_list())))
        SexRatio_inCluster = SexRatio_inCluster.div(SexRatio_inCluster.sum(axis=1), axis=0).fillna(0)[data_in[yy].cat.categories]
    reord_index=[]
    if reorder is not None:
        if any(isinstance(item, float) for item in reorder):
            l=list(enumerate(reorder))
            sorterIndex = dict(zip(sorted(reorder), range(len(reorder))))
            #reordf = pd.DataFrame(reorder)
            reord_index = [l[l[i][0]][0] for i in list(map(lambda x:sorterIndex.get(x), reorder))]
            #print(reordf)
            #reordf['my_idx'] = reordf[0].map(sorterIndex)
            #reord_index = reordf['my_idx'].to_list()
        else:
            reord_index=reorder
        SexRatio_inCluster.reset_index(drop=True, inplace=True)
        SexRatio_inCluster = SexRatio_inCluster.reindex(reord_index)
    #print(SexRatio_inCluster)
    if type(data_in) == AnnData:
        N = len(list(set(data_in.obs[xx])))
    else:
        N = len(list(set(data_in[xx]))) 
    ind = np.arange(N)    # the x locations for the groups
    width = 1
    bottom = 0
    if axo==None:
        fig, fax = plt.subplots(figsize=(N/1.6, 4.2))
    iterateover=enumerate(SexRatio_inCluster)
    try:
        ycolors=data_in.uns[str(yy)+'_colors']
    except:
        ycolors=ypalette
    for ii, d in iterateover:
        if axo != None:
            axo.bar([1*i for i in ind], SexRatio_inCluster[d] , width, bottom = bottom, color=ycolors[ii], align='edge')
        else:
            plt.bar([1*i for i in ind], SexRatio_inCluster[d] , width, bottom = bottom, color=ycolors[ii], align='edge')
        bottom += SexRatio_inCluster[d]
    try:
        xcolors = data_in.uns[str(xx)+'_colors']
        if list(xcolors).count(list(xcolors)[0]) == len(list(xcolors)):
            xcolors = xpalette
    except:
        xcolors = xpalette
    if reorder is not None:
        l=list(enumerate(xcolors))
        l = [l[l[i][0]][1] for i in reord_index]
    else:
        l = xcolors[:N]
    my_cmap = cccp.ListedColormap(l)
    font_prop = font_manager.FontProperties(size=11)
    if axo != None:
        axo.set_xticks([])
    else:
        fax.set_xticks([])
    sm = ScalarMappable(cmap=my_cmap)
    if type(data_in) == AnnData:
        sm.set_array(np.arange(4, len(data_in.obs[xx].cat.categories)+5))
    else:
        sm.set_array(np.arange(4, len(data_in[xx].cat.categories)+5))
    if axo != None:
        axo.tick_params(labelsize=8)
    else:
        fax.tick_params(labelsize=8)
    cbar = plt.colorbar(sm, orientation='horizontal', pad=0.01, aspect=4*N, ax=axo)
    cbar.set_label(xlabel, size = 11)
    if reorder is not None:
        labels = reorder
    else:
        if type(data_in) == AnnData:
            labels = data_in.obs[xx].cat.categories
        else:
            labels = data_in[xx].cat.categories
    if type(data_in) == AnnData:
        loc = np.arange(4, len(data_in.obs[xx].cat.categories)+4) + .5
    else:
        loc = np.arange(4, len(data_in[xx].cat.categories)+4) + .5
    cbar.set_ticks(loc)
    cbar.set_ticklabels(labels)
    if rotation != 0:
        rotat = 0
        if len(max(list(map(str,labels)), key=len)) > 10:
            rotat = 90
        elif len(str(labels[0])) > 3:
            rotat = 45
        else:
            rotat = 0
    else:
       rotat = rotation
    cbar.ax.tick_params(labelsize=12, color='k', direction='in', rotation=rotat)
    if axo != None:
        axo.set_ylabel(ylabel, size = 11)
        if type(data_in) == AnnData:
            axo.set_xlim(0, len(data_in.obs[xx].cat.categories))
        else:
            axo.set_xlim(0, len(data_in[xx].cat.categories))
        axo.set_ylim(0, 1)
    else:
        fax.set_ylabel(ylabel, size = 11)
        if type(data_in) == AnnData:
            fax.set_xlim(0, len(data_in.obs[xx].cat.categories))
        else:
            fax.set_xlim(0, len(data_in[xx].cat.categories))
        fax.set_ylim(0, 1)
    if legend == True:
        axo.legend(SexRatio_inCluster, ncol=1, fancybox=False, shadow=False, bbox_to_anchor=(1.003, 1.02), fontsize=8, prop=font_prop, frameon=False)# bbox_to_anchor=(1.003, 1.02)
    if ('/' in xx):
        xx=xx.replace('/','-')  
    elif ('/' in yy):
        yy=yy.replace('/','-')
    if save == True:
        plt.savefig(os.path.join(outpath,str(yy) + '_in_' + str(xx) +'_distribution.png'), bbox_inches='tight', dpi=600,format='png') 
    if axo==None:
        plt.show()

m = None

def get_n_genes(adata):
    """ return number of genes with at least 1 mapping read """
    return (adata.X>0).sum(axis=1)

def get_n_counts(adata,genes_subset=None):
    """ return number of reads mapped to genes """
    return np.sum(adata.X,axis=1)

def get_percent_mito(adata,mitochondrial_genes=None):
    """ return fraction of reads mapped to mitochondrial genes """
    if m == 'human': 
        mitochondrial_genes = adata.var_names.str.startswith('MT-')
    elif m == 'mouse':
        mitochondrial_genes = adata.var_names.str.startswith('mt-')
    mito_count,all_count = np.sum(adata[:, mitochondrial_genes].X, axis=1),get_n_counts(adata)
    return mito_count/all_count

def get_log_counts(adata):
    """ return logaritmized number of reads mapped to genes """
    return np.log(get_n_counts(adata))
    
def addQC2Adata(adata, model):
    """ compute the following QC metrics and add them to `adata`:
        - number of genes
        - total read count
        - fraction of reads mapping on mitochondrial genes """
    global m
    m = model
    n_genes,n_counts,percent_mito,log_counts = map(lambda x: x(adata), [get_n_genes,get_n_counts,get_percent_mito,get_log_counts])
    for label,value in zip(("n_genes","n_counts","percent_mito", "log_counts"), (n_genes,n_counts,percent_mito, log_counts)):
        adata.obs[label] = value

def getCountReport(adata):
    ncells = int(adata.n_obs)
    ngenes = int(adata.n_vars)
    return (ncells, ngenes)    
    
def filterData(adata,
               min_genes=3000,
               max_genes=10000,
               min_counts=100000,
               max_counts=1000000,
               mito_fract=0.2,
               min_cells=3,
               genes_to_keep=None,               
               report=False):
    """ filter data basing on QC thresholds """
    filtered = adata[(adata.obs["percent_mito"]<=mito_fract),]
    for k,v in {"min_genes":min_genes,  "max_genes":max_genes,
                "min_counts":min_counts, "max_counts":max_counts}.items():
        kwargs = {k:v}
        sc.pp.filter_cells(filtered,**kwargs)
    if genes_to_keep:
        filt_genes = sc.pp.filter_genes(filtered, min_cells=min_cells, inplace=False)
        filt_genes_df = pd.DataFrame(filt_genes).T.set_index(filtered.var_names)
        for i in genes_to_keep:
            filt_genes_df.loc[i, 0] = True
        filtered = filtered[:,filt_genes_df[0].to_list()]
        filtered.var['n_cells'] = filt_genes_df[filt_genes_df[0]==True][1].to_list()
    else:
        filt_genes = sc.pp.filter_genes(filtered, min_cells=min_cells, inplace=False)
        filtered = filtered[:,filt_genes[0].tolist()]
        filtered.var['n_cells'] = filt_genes[1][filt_genes[0]].tolist()
    if report:
        print("""
        before filtering: 
                - %s cells (%s OK, %s low quality, %s others)
                - %s genes""" %(getCountReport(adata))) 
        print("""
        after filtering:
                - %s cells (%s OK, %s low quality, %s others)
                - %s genes""" %(getCountReport(filtered)))
    return filtered

def cell_cycle_info(adata, model):
    if model not in ['human','mouse']:
        raise ValueError('The organism must be human or mouse')
        return
    org_gp = '/data2/jupyterhub/rsemeraro/old_data/HD2/Romagnani/Run_190701_NB551587_0079_AHHL75BGXB/Analyses/regev_lab_cell_cycle_genes.txt'
    cell_cycle_genes_dc = [x.strip() for x in open(org_gp)]
    s_genes = cell_cycle_genes_dc[:43]
    g2m_genes = cell_cycle_genes_dc[43:]
    if model == 'mouse':
        s_genes = [gene.lower().capitalize() for gene in s_genes]
        g2m_genes = [gene.lower().capitalize() for gene in g2m_genes]
    s_genes_mm_ens = adata.var_names[np.in1d(adata.var_names, s_genes)]
    g2m_genes_mm_ens = adata.var_names[np.in1d(adata.var_names, g2m_genes)]
    fulldata3 = adata.copy()
    cell_cycle_genes_mm = [x for x in list(s_genes_mm_ens)+list(g2m_genes_mm_ens) if x in fulldata3.var_names]
    sc.pp.scale(fulldata3)
    sc.tl.score_genes_cell_cycle(fulldata3, s_genes=s_genes_mm_ens, g2m_genes=g2m_genes_mm_ens)
    cell_cycle_genes_dc = [x for x in cell_cycle_genes_mm if x in fulldata3.var_names]
    fulldata_cc_genes = fulldata3[:, list(s_genes_mm_ens)+list(g2m_genes_mm_ens)]
    sc.tl.pca(fulldata_cc_genes, n_comps=25)
    fig = plt.figure(figsize=(14, 8.3))
    gs = fig.add_gridspec(2,3, wspace=0.5, hspace=0.3)
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[0, 2])
    ax5 = fig.add_subplot(gs[1, 2])
    sc.pl.pca_scatter(fulldata_cc_genes, color='phase', title='Non regressed out', palette=phase_cols, ax=ax1, show=False, frameon=False)
    sc.pp.regress_out(fulldata_cc_genes, ['S_score', 'G2M_score'])
    sc.tl.pca(fulldata_cc_genes)
    sc.pl.umap(fulldata_cc_genes, color=['S_score'], use_raw=False, ax=ax2, show=False, frameon=False)
    sc.pl.umap(fulldata_cc_genes, color=['G2M_score'], use_raw=False, ax=ax3, show=False, frameon=False)
    sc.pl.umap(fulldata_cc_genes, color=['phase'], use_raw=False, ax=ax4, show=False, frameon=False)
    sc.pl.pca_scatter(fulldata_cc_genes, color='phase', title='Regressed out', ax=ax5, show=False, frameon=False)
    plt.show()
    adata.obs['phase']=fulldata_cc_genes.obs['phase']
    adata.obs['S_score']=fulldata_cc_genes.obs['S_score']
    adata.obs['G2M_score']=fulldata_cc_genes.obs['G2M_score']
    adata.uns['phase_colors']=fulldata_cc_genes.uns['phase_colors']
    del fulldata3
    del fulldata_cc_genes
    return adata

def calculate_visulizations(adata, components_n=50, th=12):
    sc.pp.pca(adata, n_comps=components_n, use_highly_variable=True, svd_solver='arpack')
    sc.pp.neighbors(adata)
    sc.tl.tsne(adata, n_jobs=th)
    sc.tl.umap(adata)
    sc.tl.diffmap(adata)
    return

def addNullPoint(ad):
    """ add a null point with mit. fraction = 1 to uniform scales across dataframes """
    return ad.obs.append({'percent_mito' : 1,'n_counts':0, 'n_genes':0},
                         ignore_index=True)

def jointPlot(ad,xlim=None,ylim=None,hist_col=None,magnitude="K",palette="viridis",title=""):
    df = addNullPoint(ad)
    if xlim!=None: df = df[df['n_counts']<xlim]
    df = df[df['percent_mito']<=1]
    grid = sb.JointGrid(x='n_counts', y='n_genes', data=df)
    g = grid.plot_joint(sb.scatterplot, hue='percent_mito', data=df,palette=palette, linewidth=0,alpha=0.8)
    ax_x = sb.distplot(df['n_counts'], ax=g.ax_marg_x,color=hist_col)
    ax_y = sb.distplot(df['n_genes'], ax=g.ax_marg_y, vertical=True,color=hist_col)
    ax_x.set(xlabel='', ylabel='')
    ax_y.set(xlabel='', ylabel='')
    g.set_axis_labels(xlabel='Read counts',ylabel='Number of genes')
    if xlim==None: g.ax_marg_x.set_xlim(0,g.ax_marg_x.get_xlim()[1])
    else: g.ax_marg_x.set_xlim(0,xlim)
    if ylim==None: g.ax_marg_y.set_ylim(0,g.ax_marg_y.get_ylim()[1])
    else: g.ax_marg_y.set_ylim(0,ylim)
    if magnitude == "K": mag_n = 1000.
    else: mag_n = 1000000.
    xlabels = ['{:.1f}'.format(x) + magnitude for x in g.ax_joint.get_xticks()/mag_n]
    g.ax_joint.set_xticklabels(xlabels)
    new_labels = ["0-0.25","0.25-0.50","0.50-0.75","0.75-1"]
    handles, labels = g.ax_joint.get_legend_handles_labels()
    g.ax_joint.legend(handles=handles[1:], labels=new_labels,title="Mit. fraction")
    g.fig.suptitle(title)
    return g

def expression_level_classifier(adata, gene):
    med = np.round(np.median(adata[:,gene].X[adata[:,gene].X > 0].tolist()),1)
    adata.obs[str(gene)+'_levels'] = 'Unexpressed'
    adata.obs.loc[((adata[:,gene].X.flatten() > med).tolist()), str(gene)+'_levels'] = 'High'
    adata.obs.loc[((adata[:,gene].X.squeeze() <= med) & (adata[:,gene].X.squeeze() > 0.0)).tolist(), str(gene)+'_levels'] = 'Low'
    return adata

def sc_normalization(adata, counts=1e6, res=0.5): 
    fulldata_pp = adata.copy()
    sc.pp.normalize_per_cell(fulldata_pp, counts_per_cell_after=counts)
    sc.pp.log1p(fulldata_pp)
    sc.pp.pca(fulldata_pp, n_comps=15)
    sc.pp.neighbors(fulldata_pp)
    sc.tl.leiden(fulldata_pp, key_added='groups', resolution=res)
    input_groups = fulldata_pp.obs['groups']
    data_mat = fulldata_pp.X.T.toarray()
    from rpy2.robjects import pandas2ri
    import rpy2.robjects as robjects
    from rpy2.robjects.conversion import localconverter
    pandas2ri.activate()
    rstring="""
        function(data_mat, input_groups){
            library(scran)
            input_groups <- input_groups$groups
            size_factors <- calculateSumFactors(data_mat, clusters=input_groups, min.mean=0.1)
            return(size_factors)
        }
        """
    with localconverter(robjects.default_converter + pandas2ri.converter):
        rfunc=robjects.r(rstring)
        size_out = rfunc(pd.DataFrame(data_mat), pd.DataFrame(input_groups))
        del fulldata_pp
        adata.obs['size_factors'] = size_out
        fig = plt.figure(figsize=(13.5,4))
        gs = fig.add_gridspec(1,3, wspace=0.5)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        sc.pl.scatter(adata, 'size_factors', 'n_counts', ax=ax1, show=False, frameon=False)
        sc.pl.scatter(adata, 'size_factors', 'n_genes', ax=ax2, show=False, frameon=False)
        sb.distplot(size_out, bins=50, kde=False, ax=ax3)
        plt.show()
        adata.layers["counts"] = adata.X.copy()
        adata.X /= adata.obs['size_factors'].values[:,None]
        adata.X = scipy.sparse.csr_matrix(adata.X)
        adata.layers["norm_counts"] = adata.X.copy()
        return

def gene_group_mean(adata, adict):
    for i in adict.keys():
        ids_adata = np.in1d(adata.var_names, adict[i])
        adata.obs[i] = adata[:, ids_adata].layers['norm_log2_counts'].mean(1)
        
def scr_doublets(i, rate=0.05):
    scrub = scr.Scrublet(i.X, expected_doublet_rate=rate)
    try:
        doublet_scores, predicted_doublets = scrub.scrub_doublets(min_counts=2, 
                                                              min_cells=3, 
                                                              min_gene_variability_pctl=85, 
                                                              n_prin_comps=20, verbose=False)
        scrub.call_doublets(threshold=0.42, verbose=False)
        i.obs['Doublets'] = scrub.predicted_doublets_.tolist()
        i.obs['Doublets'] = i.obs['Doublets'].map(
                           {True:'Doublets' ,False:'Normal'})
        print('Sample doublets rate: '+str(i.obs.Doublets.value_counts('Doublets')['Doublets']))
        ii = i[i.obs['Doublets']!='Doublets'].copy()
        return ii
    except:
        pass
