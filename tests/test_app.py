import unittest
from app import app


class TestHousePriceAPI(unittest.TestCase):

    def setUp(self):

        app.config["TESTING"] = True
        self.client = app.test_client()

        self.sample_data = {
            "OverallQual": 7,
            "GrLivArea": 1800,
            "GarageCars": 2,
            "TotalBsmtSF": 900,
            "FullBath": 2,
            "YearBuilt": 2005,
            "YearRemodAdd": 2008,
            "BedroomAbvGr": 3,
            "LotArea": 8500,
            "Neighborhood": "CollgCr",
            "HouseStyle": "2Story",
            "ExterQual": "Gd"
        }

    def test_home_page(self):

        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)

    def test_metrics_endpoint(self):

        response = self.client.get("/api/metrics")

        self.assertEqual(response.status_code, 200)

        data = response.get_json()

        self.assertIn("model", data)
        self.assertIn("mae", data)
        self.assertIn("rmse", data)
        self.assertIn("r2", data)

    def test_valid_prediction(self):

        response = self.client.post(
            "/predict",
            json=self.sample_data
        )

        self.assertEqual(response.status_code, 200)

        data = response.get_json()

        self.assertTrue(data["success"])
        self.assertGreater(data["predicted_price"], 0)
        self.assertEqual(data["currency"], "USD")

    def test_missing_features(self):

        response = self.client.post(
            "/predict",
            json={"OverallQual": 7}
        )

        self.assertEqual(response.status_code, 400)

    def test_invalid_quality(self):

        invalid_data = self.sample_data.copy()
        invalid_data["OverallQual"] = 15

        response = self.client.post(
            "/predict",
            json=invalid_data
        )

        self.assertEqual(response.status_code, 400)

    def test_negative_area(self):

        invalid_data = self.sample_data.copy()
        invalid_data["GrLivArea"] = -100

        response = self.client.post(
            "/predict",
            json=invalid_data
        )

        self.assertEqual(response.status_code, 400)

    def test_invalid_json(self):

        response = self.client.post(
            "/predict",
            data="invalid data",
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
