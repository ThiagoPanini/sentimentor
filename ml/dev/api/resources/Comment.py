from flask_restful import Resource


class Comment(Resource):
    def get(self):
        return {"message": "Hello, Panini!"}