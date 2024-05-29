from src.support import clprint, Reason, get_time_in_millis


class SimpleActiveLearner:

    def __init__(self, dataset, al_technique):
        self.dataset = dataset
        self.al_technique = al_technique
        self.n_samples_to_select = -1

    def elaborate(self, model, al_epochs, training_epochs, n_samples_to_select, criterion, optimizer):
        self.n_samples_to_select = n_samples_to_select
        clprint("Starting Active Learning process...", Reason.INFO_TRAINING)
        self._train_model(criterion, model, optimizer, training_epochs)
        for i in range(al_epochs):
            clprint("Making n.{}/{} AL epochs...".format(i + 1, al_epochs), Reason.INFO_TRAINING, loggable=True)
            clprint("Selecting {} new samples...".format(self.n_samples_to_select), Reason.INFO_TRAINING)
            start_time = get_time_in_millis()
            self._select_next_samples(self.n_samples_to_select)
            end_time = get_time_in_millis()
            clprint("Elapsed time: {}".format(end_time - start_time), Reason.LIGHT_INFO_TRAINING, loggable=True)
            self._train_model(criterion, model, optimizer, training_epochs)

    def _train_model(self, criterion, model, optimizer, training_epochs):
        clprint("Training model...", Reason.INFO_TRAINING)
        model.fit(training_epochs, criterion, optimizer, self.dataset.get_train_loader())
        clprint("Evaluating model...", Reason.INFO_TRAINING)
        loss, accuracy = model.evaluate(criterion, self.dataset.get_test_loader())
        clprint("Loss: {}\nAccuracy: {}".format(loss, accuracy), Reason.LIGHT_INFO_TRAINING, loggable=True)

    def _select_next_samples(self, n_samples_to_select):
        xs = self.al_technique.select_samples(self.dataset.get_unlabeled_data(), n_samples_to_select)
        self.dataset.annotate(xs)
        clprint("Updating AL technique...".format(self.n_samples_to_select), Reason.LIGHT_INFO_TRAINING)
        self.al_technique.update(self.dataset)
