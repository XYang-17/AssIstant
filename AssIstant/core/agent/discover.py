from AssIstant.common.discoverer import Discover, Path

all_agents = Discover(Path(__file__).resolve().parent, ("Agent.py", "Agents.py")).discover("Agent")


if __name__ == "__main__":
    for a in all_agents:
        print(a.__name__)