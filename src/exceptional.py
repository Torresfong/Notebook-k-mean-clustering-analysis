class PipelineError(Exception):
    """
    Base class for all pipeline errors.
    """
    pass


class ConfigError(PipelineError):
    """
    Raised when config.yaml has missing or invalid values.
    """
    pass


class DataValidationError(PipelineError):
    """
    Raised when the raw dataset failed validation.
    """
    pass


class FeatureEngineeringError(PipelineError):
    """
    Raised when RFM feature building fails. 
    """
    pass


class ModelError(PipelineError):
    """
    Raised when model training fails or produced invalid results.
    """
    pass


class EvaluationError(PipelineError):
    """
    Raised when model evaluation fails or output saving fails.
    """
    pass


 