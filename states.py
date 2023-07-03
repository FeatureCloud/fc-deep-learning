from FeatureCloud.app.engine.app import app_state, Role, AppState

name = 'fc_deep'


def generate_federated_states(general_app_instance, **kwargs):
    from utils.pytorch.states import Initialization, LocalUpdate, GlobalAggregation, WriteResults
    @app_state(name='initial', role=Role.BOTH, app_name=name, app_instance=general_app_instance, **kwargs)
    class B1(Initialization):
        """
        Read input data
        read config files

        """

        def register(self):
            self.register_transition('Local Update', label="Broadcast initial weights")

        def run(self) -> str or None:
            super().run()
            return 'Local Update'

    @app_state('Local Update', Role.BOTH, app_instance=general_app_instance)
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

    @app_state('Global Aggregation', Role.COORDINATOR, app_instance=general_app_instance)
    class C1(GlobalAggregation):
        def register(self):
            self.register_transition('Local Update', Role.COORDINATOR, label="Broadcast the global model")
            self.register_transition('Write Results', Role.COORDINATOR, label="Finalize the execution")

        def run(self) -> str or None:
            smg = super().run()
            if smg is not None:
                return 'Write Results'
            return 'Local Update'

    @app_state('Write Results', Role.BOTH, app_instance=general_app_instance)
    class B3(WriteResults):
        def register(self):
            self.register_transition('terminal', label="Terminate the execution")

        def run(self) -> str or None:
            super().run()
            return 'terminal'

    return general_app_instance


def generate_centralized_states(app_instance, **kwargs):
    from utils.pytorch.states import Centralized
    @app_state(name='initial', app_instance=app_instance)
    class S1(AppState):
        """
        Read input data
        read config files

        """

        def register(self):
            self.register_transition('Centralized Training', role=Role.COORDINATOR, label="Local state transition")

        def run(self) -> str or None:
            return 'Centralized Training'

    @app_state(name='Centralized Training', role=Role.COORDINATOR, app_instance=app_instance, app_name=name, **kwargs)
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

    return app_instance


# def generate_simulation_states(app_instance, **kwargs):
#     from utils.pytorch.states import Simulation
#     @app_state(name='initial', role=Role.BOTH, app_instance=app_instance)
#     class S1(AppState):
#         """
#         Read input data
#         read config files
#
#         """
#
#         def register(self):
#             self.register_transition('Federated Simulation', role=Role.COORDINATOR, label="Local state transition")
#
#         def run(self) -> str or None:
#             return 'Federated Simulation'
#
#     @app_state(name='Federated Simulation', role=Role.COORDINATOR, app_name=name, app_instance=app_instance, **kwargs)
#     class FederatedSimulation(Simulation):
#         """
#         Read input data
#         read config files
#         """
#
#         def register(self):
#             self.register_transition('terminal', role=Role.COORDINATOR, label="Terminate the execution")
#
#         def run(self) -> str or None:
#             super().run()
#             return 'terminal'
#
#     return app_instance
