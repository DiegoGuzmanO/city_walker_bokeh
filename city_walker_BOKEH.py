
import streamlit as st
st.set_option(
    'deprecation.showPyplotGlobalUse', 
    False
    )

st.image(
    'page_de_garde.png', 
         caption=None, 
         width=None, 
         use_column_width=False, 
         clamp=False, 
         channels='RGB', 
         output_format='auto'
         )

code_display = st.sidebar.radio(label="Afficher le code :", 
                                options=["Non", "Oui"])

if code_display =="Oui":
    with st.echo():


        import pandas as pd
        import numpy as np
        from sklearn.cluster import KMeans
        from scipy.spatial.distance import cdist
        
        from sklearn.cluster import SpectralClustering
        import networkx as nx
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.cluster import OPTICS
        #import des modules bokeh
        from bokeh.plotting import figure
        from bokeh.models import HoverTool, ColumnDataSource
        from bokeh.tile_providers import get_provider
        
        tuile=get_provider('CARTODBPOSITRON_RETINA')
        import warnings
        warnings.filterwarnings("ignore")
        
        
        # Lecture du dataframe nettoyé
        df_ville = pd.read_csv(
            'df_paris_bokeh.csv',
            sep = ';'
            )
        
        #Conversion des données epsg:4326 en espg:3857
        ##Conversion déjà faite dans le csv de départ. Pour un nv fichier, appliquer
        ##le code ci-dessous
        #k=6378137
        #df_ville['X'] = df_ville['X'].apply(lambda x: x*(k * np.pi / 180.0))
        #df_ville['Y'] = df_ville['Y'].apply(lambda x: np.log(np.tan((90 + x) * np.pi / 360.0)) * k)
        
        #######################################################################
        # Sélection des paramètres utilisateur
        
        ## Nombre de jours
        st.sidebar.title(
            "Planification de l'itinéraire"
            ) 
        
        slider_clusters = st.sidebar.slider(
            label='Nombre de jours à Paris :',
                                    min_value=1,
                                    max_value=15,
                                    value=3,
                                    step=1)
        
        ## Modèle du classificateur
        if slider_clusters <11 : 
            modele_optimal = 'Spectral Clustering'
        else : 
            modele_optimal = 'K-Means'
        
        st.sidebar.text('Modèle recommandé :')
        
        st.sidebar.text(modele_optimal)
        
        slider_modele = st.sidebar.select_slider(
            label='Confirmez le modèle souhaité :',
            options=['Spectral Clustering',
                     'K-Means'],value= modele_optimal
            )
        
        ## Nombre de restaurants à proposer
        slider_nb_restos=st.sidebar.slider(
            label='Nombre de restaurants proposés :',
                                           min_value=1,
                                           max_value=15,
                                           value=3,
                                           step=1
                                           )
        
        ##Type de restaurants à afficher
        selectbox_type_restaurant = st.sidebar.selectbox(
            label='Type de restaurants :',
            options=['Peu importe', 'Restaurant traditionnel','Fast Food']
            )
        
        ############################################################################
        
        #Création des subsets par catégorie de lieu
        ##Catégorie Patrimoine
        df_patrimoine=df_ville[
            (df_ville['categorie']=='patrimoine')
            ]
        
        ##Catégorie Shopping
        df_shopping=df_ville[ 
            (df_ville['categorie']=='shopping')
            ]
        
        ##Catégorie Restaurant
        if selectbox_type_restaurant=='Peu importe':
            df_restaurants=df_ville[df_ville['categorie']=='restaurant']
            
        elif selectbox_type_restaurant=='Restaurant traditionnel':
            df_restaurants=df_ville[
                (df_ville['categorie']=='restaurant')
                & (df_ville['type']=='restaurant traditionnel')
                ]
            
        else:
            df_restaurants=df_ville[
                (df_ville['categorie']=='restaurant')
                & (df_ville['type']=='fast_food')]
        
        #######################################################################
        #Boucle de sélection du modèle de clusering
        
        if modele_optimal == 'Spectral Clustering':
            # Specral Clustering avec n_clusters 15
            spectral1 = SpectralClustering(
                n_clusters=15,
                affinity = 'nearest_neighbors'
                ).fit(df_patrimoine.loc[:,['X','Y']])
            labelsspec1=spectral1.labels_
            df_patrimoine['label']=labelsspec1
            
        else:
            # modèle Kmeans avec n_clusters = 15
            kmeans=KMeans(
                n_clusters=15
                ).fit(df_patrimoine.loc[:,['X','Y']])
            labels_k=kmeans.labels_
            df_patrimoine['label']=labels_k
            
        
        #######################################################################
        # Identification des points prioritaires PageRank par cluster 
        tops_per_cluster=pd.DataFrame()
        tops_coord=pd.DataFrame()
        
        for i in range(len(df_patrimoine['label'].unique())):
          X = df_patrimoine[df_patrimoine['label']==i]['X'].to_numpy().reshape(-1,1)
          Y = df_patrimoine[df_patrimoine['label']==i]['Y'].to_numpy().reshape(-1,1)
        
          XY = np.concatenate(
              [X, Y],
              axis = 1
              )
        
        
        ## Matrice des distances
          distances = cdist(
              XY, 
              XY, 
              metric = 'euclidean'
              )
        
        
        ## Scaling entre 0 et 1
          normalized_dists = MinMaxScaler().fit_transform(distances)
        
        ## On inverse les distances pour que les lieux les mieux classés
        ## soient les lieux les plus proches des autres
          normalized_dists = 1 - normalized_dists
        
        ## On ne veut pas qu'un point soit fortement connecté avec lui même
          normalized_dists = normalized_dists - np.eye(len(X))
        
        ## Normalisation des distances pour obtenir des valeurs probabilistiques
          normalized_dists /= normalized_dists.sum(axis = 1).reshape(-1, 1)
        
        ## Application du pagerank
          G = nx.from_numpy_matrix(normalized_dists) 
          rankings = nx.pagerank(G)
        
        ## Top nodes du cluster
          top_nodes = sorted(
              rankings.items(), 
              key = lambda x: x[1], 
              reverse = True
              )[:5]
        
        ## Enregistrement des coordonnées des top_nodes dans un dataframe "tops_coord"
          coord=[]
          for top in top_nodes:
            coord.append(XY[top[0]])
          tops_coord[i]=coord
        
        
        ## Enregistrement des top nodes dans le dataframe recapitulatif
          tops_per_cluster[i]=(top_nodes)
        
        
        ## Calcul et enregistrement des centroids des pagerank par cluster dans un dataframe "top_centroids"
        
        top_centroids=pd.DataFrame(index=('X','Y'))
        for cluster in tops_coord.columns:
          top_centroids[cluster]=tops_coord[cluster].mean()
        top_centroids=top_centroids.transpose()
        
        #######################################################################
        #Traitement des restaurants
        
        df_restaurants.reset_index(inplace=True)
        Xr = df_restaurants['X'].to_numpy().reshape(-1, 1)
        Yr = df_restaurants['Y'].to_numpy().reshape(-1, 1)
        XYr=np.concatenate([Xr,Yr], axis = 1)
        
        ## Identification des restaurants situés le plus près des top_centroids
        restos_index=[]
        for cluster in top_centroids.index:
          xy=np.array(
              [top_centroids.iloc[cluster,0],
               top_centroids.iloc[cluster,1]]
              )
          restos=cdist(
              XYr,
              [xy],
              metric='euclidean'
              )
          dist_df=pd.DataFrame(restos)
          liste=dist_df.sort_values(by=0).head(slider_nb_restos).index.tolist()
          restos_index.append(liste)
        
        #######################################################################
        # Identification des zones commerciales par l'algorithme OPTICS de ScikitLearn
        optics_clf = OPTICS(
            min_samples=15,
            metric='euclidean',
            cluster_method='xi'
            ).fit(df_shopping.loc[:,['X','Y']])
        shop_labels = optics_clf.labels_
        
        ## création des polygones à partir des clusters issues du modèle optics_clf
        df_shopping['label']=shop_labels
        
        ## Suppression du cluster -1 qui correspond aux points "noise"
        df_shopping=df_shopping[df_shopping['label'] > -1]
        df_shopping=df_shopping.reset_index()
        
        # Création d'un DataFrame qui enregistrera les coordonnées des centroids 
        # théoriques des zones commerciales
        shop_centroids=pd.DataFrame(columns=['cluster','X','Y'])
        
        # Initialisation de la boucle qui va tracer chaque patch de zone commerciale
        for cluster in df_shopping['label'].sort_values().unique():
        
        # Création d'un dataframe qui va contenir les coordonnées du patch
        
          coords_patchs=pd.DataFrame()
        
          # création de listes intermédiaires qui vont enregistrer les coordonnées du patch
          coords_y_list=[]
          coords_x_list=[]
          
          for i,z in zip(
                  df_shopping[df_shopping.label==cluster]['Y'],
                  df_shopping[df_shopping.label==cluster]['X']
                  ):
            coords_y_list.append(i)
            coords_x_list.append(z)
        
          # Enregistrement de chaque liste de coordonnées dans le dataframe central
          coords_patchs['Y']=coords_y_list
          coords_patchs['X']=coords_x_list
        
          #enregistrement du centroid de la zone commerciale
          cluster_centroid=pd.DataFrame(
              [[cluster,coords_patchs['X'].mean(),coords_patchs['Y'].mean()]],
              columns=['cluster','X','Y']
              )
          shop_centroids=shop_centroids.append(cluster_centroid)
          shop_centroids=shop_centroids.reset_index().drop(columns=['index'])
        
        ## identification des zones de Shopping les plus près des top_centroids
        
        Xs = shop_centroids['X'].to_numpy().reshape(-1, 1)
        Ys = shop_centroids['Y'].to_numpy().reshape(-1, 1)
        XYs=np.concatenate([Xs,Ys], axis = 1)
        
        shop_centroids_per_cluster=[]
        for cluster in top_centroids.index:
          xy=np.array(
              [top_centroids.iloc[cluster,0],
               top_centroids.iloc[cluster,1]]
              )
          shop_dist=cdist(
              XYs,
              [xy],
              metric='euclidean'
              )
          shop_dist=pd.DataFrame(shop_dist)
          liste=shop_dist.sort_values(by=0).head(3).index.tolist()
          shop_centroids_per_cluster.append(liste)
        
        
        
        #######################################################################
        
        
        
        # Affichage des top 5 pagerank, restaurants et zones commerciales
        
        for cluster in range(slider_clusters):
            
            # Création de la figure à ploter
            jour=cluster+1
            p=figure(plot_width=600,
                     plot_height=400,
                     x_range = (250598, 269950),
                     y_range = (6242153, 6259275), 
                     title= 'jour de visite n°%i' %jour)
            tuile=get_provider('CARTODBPOSITRON_RETINA')
            
            p.axis.visible=False
            p.add_tile(tuile)   
            
            
            # préparation des couleurs des points prio et enregistrement de leurs coord
            line_coords=pd.DataFrame(columns=('X','Y'))
            colors = ['blue' for i in range(120)]
            
            for node in tops_per_cluster.iloc[:,cluster]:
                colors[node[0]] = 'red'
                line_coords=line_coords.append(
                    {"X":df_patrimoine[df_patrimoine['label']==cluster].iloc[node[0],1],
                     'Y':df_patrimoine[df_patrimoine['label']==cluster].iloc[node[0],2]},
                    ignore_index=True
                    )
                
        
            #Affichage des points patrimoine
            df_patrimoine_cluster=df_patrimoine[df_patrimoine['label']==cluster]
            colors=colors[0:len(df_patrimoine_cluster)]
            df_patrimoine_cluster['colors']=colors
            source_patrimoine=ColumnDataSource(df_patrimoine_cluster)
            renderer1=p.circle(
                x='X',
                y='Y',
                size=8,
                alpha=0.7,
                source=source_patrimoine,
                color='colors'
                )
            
            #Affichage des restaurants
            source_restaurants=ColumnDataSource(df_restaurants.loc[restos_index[cluster]])
            renderer2=p.circle(
                x='X',
                y='Y',
                size=8,
                alpha=0.7,
                source=source_restaurants,
                color='green')
            
            
            #Affichage des zones commerciales
            for zone in shop_centroids_per_cluster[cluster]:
                source_shopping=ColumnDataSource(df_shopping[df_shopping.label==zone])
                p.patch(
                    x='X',
                    y='Y',
                    source=source_shopping,
                    color='orange',
                    alpha=0.5
                    )
            
            # Optimisation du trajet entre points prioritaires
            ##Calcule des distances
            distances = cdist(
                line_coords, 
                line_coords, 
                metric = 'euclidean'
                )
            
            ## Scaling entre 0 et 1
            normalized_dists = MinMaxScaler().fit_transform(distances)
            
            ## On ne veut pas qu'un point soit fortement connecté avec lui même
            normalized_dists = normalized_dists - np.eye(len(line_coords))
            
            ## Application du Minimum Spanning Tree pour identifier le trajet optimal
            G = nx.from_numpy_matrix(normalized_dists) 
            mst=nx.tree.minimum_spanning_tree(G)
            order=[*mst.edges]
            
            
            ##Tracé du parcours optimal
            x_coords=[]
            y_coords=[]
            for edge in range(len(order)):
                x_coords.append(
                    [line_coords[line_coords.index==order[edge][0]]['X'],
                     line_coords[line_coords.index==order[edge][1]]['X']]
                    )
                y_coords.append(
                    [line_coords[line_coords.index==order[edge][0]]['Y'],
                     line_coords[line_coords.index==order[edge][1]]['Y']]
                    )
        
            p.multi_line(
                xs=x_coords,
                ys=y_coords,
                color='red',
                line_dash='dashed',
                line_width=4
                )
        
            tooltips=[
                ('Nom','@name'),
                ('Catégorie','@categorie'),
                ('Type','@type')
                ]
            h=HoverTool(
                renderers=[renderer1,renderer2],
                tooltips=tooltips
                )
            
            p.add_tools(h)
                
            st.bokeh_chart(p)

else:
    
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    
    from sklearn.cluster import SpectralClustering
    import networkx as nx
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import OPTICS
    #import des modules bokeh
    from bokeh.plotting import figure
    from bokeh.models import HoverTool, ColumnDataSource
    from bokeh.tile_providers import get_provider
    
    tuile=get_provider('CARTODBPOSITRON_RETINA')
    import warnings
    warnings.filterwarnings("ignore")
    
    
    # Lecture du dataframe nettoyé
    df_ville = pd.read_csv(
        'df_paris_bokeh.csv',
        sep = ';'
        )
    
    #Conversion des données epsg:4326 en espg:3857
    ##Conversion déjà faite dans le csv de départ. Pour un nv fichier, appliquer
    ##le code ci-dessous
    #k=6378137
    #df_ville['X'] = df_ville['X'].apply(lambda x: x*(k * np.pi / 180.0))
    #df_ville['Y'] = df_ville['Y'].apply(lambda x: np.log(np.tan((90 + x) * np.pi / 360.0)) * k)
    
    #######################################################################
    # Sélection des paramètres utilisateur
    
    ## Nombre de jours
    st.sidebar.title(
        "Planification de l'itinéraire"
        ) 
    
    slider_clusters = st.sidebar.slider(
        label='Nombre de jours à Paris :',
                                min_value=1,
                                max_value=15,
                                value=3,
                                step=1)
    
    ## Modèle du classificateur
    if slider_clusters <11 : 
        modele_optimal = 'Spectral Clustering'
    else : 
        modele_optimal = 'K-Means'
    
    st.sidebar.text('Modèle recommandé :')
    
    st.sidebar.text(modele_optimal)
    
    slider_modele = st.sidebar.select_slider(
        label='Confirmez le modèle souhaité :',
        options=['Spectral Clustering',
                 'K-Means'],value= modele_optimal
        )
    
    ## Nombre de restaurants à proposer
    slider_nb_restos=st.sidebar.slider(
        label='Nombre de restaurants proposés :',
                                       min_value=1,
                                       max_value=15,
                                       value=3,
                                       step=1
                                       )
    
    ##Type de restaurants à afficher
    selectbox_type_restaurant = st.sidebar.selectbox(
        label='Type de restaurants :',
        options=['Peu importe', 'Restaurant traditionnel','Fast Food']
        )
    
    ############################################################################
    
    #Création des subsets par catégorie de lieu
    ##Catégorie Patrimoine
    df_patrimoine=df_ville[
        (df_ville['categorie']=='patrimoine')
        ]
    
    ##Catégorie Shopping
    df_shopping=df_ville[ 
        (df_ville['categorie']=='shopping')
        ]
    
    ##Catégorie Restaurant
    if selectbox_type_restaurant=='Peu importe':
        df_restaurants=df_ville[df_ville['categorie']=='restaurant']
        
    elif selectbox_type_restaurant=='Restaurant traditionnel':
        df_restaurants=df_ville[
            (df_ville['categorie']=='restaurant')
            & (df_ville['type']=='restaurant')
            ]
        
    else:
        df_restaurants=df_ville[
            (df_ville['categorie']=='restaurant')
            & (df_ville['type']=='fast_food')]
    
    #######################################################################
    #Boucle de sélection du modèle de clusering
    
    if modele_optimal == 'Spectral Clustering':
        # Specral Clustering avec n_clusters 15
        spectral1 = SpectralClustering(
            n_clusters=15,
            affinity = 'nearest_neighbors'
            ).fit(df_patrimoine.loc[:,['X','Y']])
        labelsspec1=spectral1.labels_
        df_patrimoine['label']=labelsspec1
        
    else:
        # modèle Kmeans avec n_clusters = 15
        kmeans=KMeans(
            n_clusters=15
            ).fit(df_patrimoine.loc[:,['X','Y']])
        labels_k=kmeans.labels_
        df_patrimoine['label']=labels_k
        
    
    #######################################################################
    # Identification des points prioritaires PageRank par cluster 
    tops_per_cluster=pd.DataFrame()
    tops_coord=pd.DataFrame()
    
    for i in range(len(df_patrimoine['label'].unique())):
      X = df_patrimoine[df_patrimoine['label']==i]['X'].to_numpy().reshape(-1,1)
      Y = df_patrimoine[df_patrimoine['label']==i]['Y'].to_numpy().reshape(-1,1)
    
      XY = np.concatenate(
          [X, Y],
          axis = 1
          )
    
    
    ## Matrice des distances
      distances = cdist(
          XY, 
          XY, 
          metric = 'euclidean'
          )
    
    
    ## Scaling entre 0 et 1
      normalized_dists = MinMaxScaler().fit_transform(distances)
    
    ## On inverse les distances pour que les lieux les mieux classés
    ## soient les lieux les plus proches des autres
      normalized_dists = 1 - normalized_dists
    
    ## On ne veut pas qu'un point soit fortement connecté avec lui même
      normalized_dists = normalized_dists - np.eye(len(X))
    
    ## Normalisation des distances pour obtenir des valeurs probabilistiques
      normalized_dists /= normalized_dists.sum(axis = 1).reshape(-1, 1)
    
    ## Application du pagerank
      G = nx.from_numpy_matrix(normalized_dists) 
      rankings = nx.pagerank(G)
    
    ## Top nodes du cluster
      top_nodes = sorted(
          rankings.items(), 
          key = lambda x: x[1], 
          reverse = True
          )[:5]
    
    ## Enregistrement des coordonnées des top_nodes dans un dataframe "tops_coord"
      coord=[]
      for top in top_nodes:
        coord.append(XY[top[0]])
      tops_coord[i]=coord
    
    
    ## Enregistrement des top nodes dans le dataframe recapitulatif
      tops_per_cluster[i]=(top_nodes)
    
    
    ## Calcul et enregistrement des centroids des pagerank par cluster dans un dataframe "top_centroids"
    
    top_centroids=pd.DataFrame(index=('X','Y'))
    for cluster in tops_coord.columns:
      top_centroids[cluster]=tops_coord[cluster].mean()
    top_centroids=top_centroids.transpose()
    
    #######################################################################
    #Traitement des restaurants
    
    df_restaurants.reset_index(inplace=True)
    Xr = df_restaurants['X'].to_numpy().reshape(-1, 1)
    Yr = df_restaurants['Y'].to_numpy().reshape(-1, 1)
    XYr=np.concatenate([Xr,Yr], axis = 1)
    
    ## Identification des restaurants situés le plus près des top_centroids
    restos_index=[]
    for cluster in top_centroids.index:
      xy=np.array(
          [top_centroids.iloc[cluster,0],
           top_centroids.iloc[cluster,1]]
          )
      restos=cdist(
          XYr,
          [xy],
          metric='euclidean'
          )
      dist_df=pd.DataFrame(restos)
      liste=dist_df.sort_values(by=0).head(slider_nb_restos).index.tolist()
      restos_index.append(liste)
    
    #######################################################################
    # Identification des zones commerciales par l'algorithme OPTICS de ScikitLearn
    optics_clf = OPTICS(
        min_samples=15,
        metric='euclidean',
        cluster_method='xi'
        ).fit(df_shopping.loc[:,['X','Y']])
    shop_labels = optics_clf.labels_
    
    ## création des polygones à partir des clusters issues du modèle optics_clf
    df_shopping['label']=shop_labels
    
    ## Suppression du cluster -1 qui correspond aux points "noise"
    df_shopping=df_shopping[df_shopping['label'] > -1]
    df_shopping=df_shopping.reset_index()
    
    # Création d'un DataFrame qui enregistrera les coordonnées des centroids 
    # théoriques des zones commerciales
    shop_centroids=pd.DataFrame(columns=['cluster','X','Y'])
    
    # Initialisation de la boucle qui va tracer chaque patch de zone commerciale
    for cluster in df_shopping['label'].sort_values().unique():
    
    # Création d'un dataframe qui va contenir les coordonnées du patch
    
      coords_patchs=pd.DataFrame()
    
      # création de listes intermédiaires qui vont enregistrer les coordonnées du patch
      coords_y_list=[]
      coords_x_list=[]
      
      for i,z in zip(
              df_shopping[df_shopping.label==cluster]['Y'],
              df_shopping[df_shopping.label==cluster]['X']
              ):
        coords_y_list.append(i)
        coords_x_list.append(z)
    
      # Enregistrement de chaque liste de coordonnées dans le dataframe central
      coords_patchs['Y']=coords_y_list
      coords_patchs['X']=coords_x_list
    
      #enregistrement du centroid de la zone commerciale
      cluster_centroid=pd.DataFrame(
          [[cluster,coords_patchs['X'].mean(),coords_patchs['Y'].mean()]],
          columns=['cluster','X','Y']
          )
      shop_centroids=shop_centroids.append(cluster_centroid)
      shop_centroids=shop_centroids.reset_index().drop(columns=['index'])
    
    ## identification des zones de Shopping les plus près des top_centroids
    
    Xs = shop_centroids['X'].to_numpy().reshape(-1, 1)
    Ys = shop_centroids['Y'].to_numpy().reshape(-1, 1)
    XYs=np.concatenate([Xs,Ys], axis = 1)
    
    shop_centroids_per_cluster=[]
    for cluster in top_centroids.index:
      xy=np.array(
          [top_centroids.iloc[cluster,0],
           top_centroids.iloc[cluster,1]]
          )
      shop_dist=cdist(
          XYs,
          [xy],
          metric='euclidean'
          )
      shop_dist=pd.DataFrame(shop_dist)
      liste=shop_dist.sort_values(by=0).head(3).index.tolist()
      shop_centroids_per_cluster.append(liste)
    
    
    
    #######################################################################
    
    
    
    # Affichage des top 5 pagerank, restaurants et zones commerciales
    
    for cluster in range(slider_clusters):
        
        # Création de la figure à ploter
        jour=cluster+1
        p=figure(plot_width=600,
                 plot_height=400,
                 x_range = (250598, 269950),
                 y_range = (6242153, 6259275), 
                 title= 'jour de visite n°%i' %jour)
        tuile=get_provider('CARTODBPOSITRON_RETINA')
        
        p.axis.visible=False
        p.add_tile(tuile)   
        
        
        # préparation des couleurs des points prio et enregistrement de leurs coord
        line_coords=pd.DataFrame(columns=('X','Y'))
        colors = ['blue' for i in range(120)]
        
        for node in tops_per_cluster.iloc[:,cluster]:
            colors[node[0]] = 'red'
            line_coords=line_coords.append(
                {"X":df_patrimoine[df_patrimoine['label']==cluster].iloc[node[0],1],
                 'Y':df_patrimoine[df_patrimoine['label']==cluster].iloc[node[0],2]},
                ignore_index=True
                )
            
    
        #Affichage des points patrimoine
        df_patrimoine_cluster=df_patrimoine[df_patrimoine['label']==cluster]
        colors=colors[0:len(df_patrimoine_cluster)]
        df_patrimoine_cluster['colors']=colors
        source_patrimoine=ColumnDataSource(df_patrimoine_cluster)
        renderer1=p.circle(
            x='X',
            y='Y',
            size=8,
            alpha=0.7,
            source=source_patrimoine,
            color='colors'
            )
        
        #Affichage des restaurants
        source_restaurants=ColumnDataSource(df_restaurants.loc[restos_index[cluster]])
        renderer2=p.circle(
            x='X',
            y='Y',
            size=8,
            alpha=0.7,
            source=source_restaurants,
            color='green')
        
        
        #Affichage des zones commerciales
        for zone in shop_centroids_per_cluster[cluster]:
            source_shopping=ColumnDataSource(df_shopping[df_shopping.label==zone])
            p.patch(
                x='X',
                y='Y',
                source=source_shopping,
                color='orange',
                alpha=0.5
                )
        
        # Optimisation du trajet entre points prioritaires
        ##Calcule des distances
        distances = cdist(
            line_coords, 
            line_coords, 
            metric = 'euclidean'
            )
        
        ## Scaling entre 0 et 1
        normalized_dists = MinMaxScaler().fit_transform(distances)
        
        ## On ne veut pas qu'un point soit fortement connecté avec lui même
        normalized_dists = normalized_dists - np.eye(len(line_coords))
        
        ## Application du Minimum Spanning Tree pour identifier le trajet optimal
        G = nx.from_numpy_matrix(normalized_dists) 
        mst=nx.tree.minimum_spanning_tree(G)
        order=[*mst.edges]
        
        
        ##Tracé du parcours optimal
        x_coords=[]
        y_coords=[]
        for edge in range(len(order)):
            x_coords.append(
                [line_coords[line_coords.index==order[edge][0]]['X'],
                 line_coords[line_coords.index==order[edge][1]]['X']]
                )
            y_coords.append(
                [line_coords[line_coords.index==order[edge][0]]['Y'],
                 line_coords[line_coords.index==order[edge][1]]['Y']]
                )
    
        p.multi_line(
            xs=x_coords,
            ys=y_coords,
            color='red',
            line_dash='dashed',
            line_width=4
            )
    
        tooltips=[
            ('Nom','@name'),
            ('Catégorie','@categorie'),
            ('Type','@type')
            ]
        h=HoverTool(
            renderers=[renderer1,renderer2],
            tooltips=tooltips
            )
        
        p.add_tools(h)
            
        st.bokeh_chart(p)
