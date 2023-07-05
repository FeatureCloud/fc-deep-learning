import app_four_utils as utils


if __name__ == '__main__':
    if utils.is_native():
        print("The app will run in Native mode...")
        if utils.is_centralized():
            print("Centralized analysis...")
            utils.centralized()
        elif utils.is_simulation():
            print("Simulating federated analysis...")
            utils.simulate()
        else:
            raise NotImplemented(f"Native execution is only available for 'centralized' or `simulation` scenarios")
    else:
        if utils.is_centralized():
            utils.centralized()
        elif utils.is_simulation():
            utils.simulate()
        else:
            utils.federated()
