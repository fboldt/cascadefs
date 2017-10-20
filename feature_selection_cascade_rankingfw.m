function cascade = feature_selection_cascade_rankingfw(fscriterion)
cascade = feature_selection_cascade();
cascade = cascade.add_feature_selection_method(cascade, feature_selection_ranking(fscriterion_filter));
cascade = cascade.add_feature_selection_method(cascade, feature_selection_ranking(fscriterion));
end

