from flask import jsonify


class Result:
    """
    统一的返回类
    """
    def __init__(self, success, code, message):
        self.success = success
        self.code = code
        self.message = message

    def to_dict(self):
        return {
            "success": self.success,
            "code": self.code,
            "message": self.message,
        }

    @classmethod
    def ok(cls, data=None):
        message = '成功' if data is None else data
        result = cls(success=True, code=200, message=message)
        return jsonify(result.to_dict())

    @classmethod
    def error(cls, msg):
        result = cls(success=False, code=500, message=msg)
        return jsonify(result.to_dict())