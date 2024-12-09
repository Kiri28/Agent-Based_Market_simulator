from abc import ABC, abstractmethod

# Base agent logic
class BaseAgent(ABC):
	#agent_id: Optional[int] = None

	@abstractmethod
	def get_statistics(self):
		pass

	@abstractmethod
	def trading_logic(self):
		pass

	@abstractmethod
	def trading_step(self):
		pass