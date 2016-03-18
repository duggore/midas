
NCLUSTERS=$1

echo "Running cluster ml for N clusters: "$NCLUSTERS
echo "expanding"
shopt -s expand_aliases
echo "sourcing"
source ~/.bashrc

#/net/zeta/zeta/home/op251/Documents/2year/modified_snap/examples/bigclam/bigclam -i:/mallow/data/2year/mito_graph/midas_input_data/edge_thresholded_human_string_graphs/barabasi_edges/scores_removed_barabasi_graph.txt -c:200 -nt:8

bigclam="/home/op251/Documents/2year/modified_snap/examples/bigclam/bigclam"
#graph_fname="/mallow/data/2year/mito_graph/midas_input_data/edge_thresholded_human_string_graphs/barabasi_edges/scores_removed_barabasi_graph.txt"
#graph_fname="/mallow/data/2year/mito_graph/midas_input_data/edge_thresholded_human_string_graphs/all_edges/0.0_threshold/scores_removed_threshold_0.0_human_only_string.txt"
#graph_fname="/mallow/data/2year/mito_graph/v10_raw_string_downloads/no_scores_string_v10_human_links.txt"

#graph_fname="/mallow/data/2year/mito_graph/v10_raw_string_downloads/edge_thresholded_human_string_graphs/0.4_threshold/scores_removed_threshold_0.4_human_protein.links.v10.txt"

graph_fname="/mallow/data/2year/mito_graph/v10_raw_string_downloads/no_scores_non_textmining_edges_9606.protein.links.detailed.v10.txt"

reformat_communities_py="/mallow/data/2year/mito_graph/midas_input_data/edge_thresholded_human_string_graphs/barabasi_edges/data_processing/reformat_communities_from_snap_output.py"

echo "Running BIGCLAM"
$bigclam -i:$graph_fname -c:$NCLUSTERS -nt:8

#communities_fname="/mallow/data/2year/mito_graph/midas_input_data/edge_thresholded_human_string_graphs/all_edges/0.0_threshold/string_reformatted_"$NCLUSTERS"_communities.txt"
#communities_fname="/mallow/data/2year/mito_graph/v10_raw_string_downloads/clustering/string_v10_0.4_threshold_reformatted_"$NCLUSTERS"_communities.txt"
communities_fname="/mallow/data/2year/mito_graph/v10_raw_string_downloads/clustering/string_v10_no_textmining_reformatted_"$NCLUSTERS"_communities.txt"
python $reformat_communities_py cmtyvv.txt $communities_fname

echo "Starting Machine learning stage"
python ml_pipeline_graph_clusters.py -communities_filename=$communities_fname -run_desc="string_"$NCLUSTERS"_communities/"
echo "finished"
