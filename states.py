from utils.pytorch.states import Initialization, LocalUpdate, GlobalAggregation, WriteResults, Centralized
from FeatureCloud.app.engine.app import app_state, Role

name = 'fc_deep'


@app_state(name='initial', role=Role.BOTH, app_name=name)
class B1(Initialization):
    """
    Read input data
    read config files

    """

    def register(self):
        self.register_transition('Local Update', label="Broadcast initial weights")
        self.register_transition('Centralized Training', role=Role.COORDINATOR, label="Local state transition")

    def run(self) -> str or None:
        super().run()
        if self.config['centralized'] and len(self.clients) == 1 and self.coordinator:
            return 'Centralized Training'
        return 'Local Update'


@app_state('Local Update', Role.BOTH)
class B2(LocalUpdate):
    """ Local Model training
        Input:
            Model weights(Coordinator already has it)
            App statuses: {Converged: True/False }
    """

    def register(self):
        self.register_transition('Global Aggregation', Role.COORDINATOR, label="Gather local models")
        self.register_transition('Local Update', Role.PARTICIPANT, label="Wait for the global model")
        self.register_transition('Write Results', Role.PARTICIPANT, label="Finalize the execution")

    def run(self) -> str or None:

        msg = super().run()
        if msg is not None:
            return 'Write Results'
        if self.is_coordinator:
            return "Global Aggregation"
        return 'Local Update'


@app_state('Global Aggregation', Role.COORDINATOR)
class C1(GlobalAggregation):
    def register(self):
        self.register_transition('Local Update', Role.COORDINATOR, label="Broadcast the global model")
        self.register_transition('Write Results', Role.COORDINATOR, label="Finalize the execution")

    def run(self) -> str or None:
        smg = super().run()
        if smg is not None:
            return 'Write Results'
        return 'Local Update'


@app_state('Write Results', Role.BOTH)
class B3(WriteResults):
    def register(self):
        self.register_transition('terminal', label="Terminate the execution")

    def run(self) -> str or None:
        super().run()
        return 'terminal'


@app_state(name='Centralized Training', role=Role.COORDINATOR)
class CentralizedTraining(Centralized):
    """
    Read input data
    read config files

    """

    def register(self):
        self.register_transition('terminal', role=Role.COORDINATOR, label="Terminate the execution")

    def run(self) -> str or None:
        super().run()
        return 'terminal'
