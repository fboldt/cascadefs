function cascade = feature_selection_cascade_hybridranking_hybridsfs(fscriterion)
cascade = feature_selection_cascade();
cascade = cascade.add_feature_selection_method(cascade, feature_selection_hybridranking(fscriterion));
cascade = cascade.add_feature_selection_method(cascade, feature_selection_hybridsfs(fscriterion));
end

