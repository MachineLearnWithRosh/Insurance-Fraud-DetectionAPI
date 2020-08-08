from datetime import datetime
import pandas as pd
from flask import jsonify, Response


class dataTransformation:
    def __init__(self, data):
        self.data = data

    def map_categorical_data(self):
        try:
            raw_dict = self.data
            self.cols = ['months_as_customer', 'age', 'policy_state', 'policy_csl', 'policy_deductable',
                    'policy_annual_premium',
                    'umbrella_limit', 'insured_sex', 'insured_education_level', 'insured_occupation', 'insured_hobbies',
                    'insured_relationship', 'capital-gains', 'capital-loss', 'incident_type', 'collision_type',
                    'incident_severity', 'authorities_contacted',
                    'incident_state', 'incident_city', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
                    'property_damage', 'bodily_injuries',
                    'witnesses', 'police_report_available', 'total_claim_amount', 'injury_claim', 'property_claim',
                    'vehicle_claim',
                    'auto_make', 'auto_model', 'auto_year']
            cols = self.cols

            for item in cols:
                if item not in raw_dict:
                    print(item)
                    return "Invalid Input"

            curated_dict = {i: raw_dict[i] for i in raw_dict if i in cols}
            df = pd.DataFrame([curated_dict], columns=curated_dict.keys())
            df['policy_state'] = df['policy_state'].map({'OH': 2, 'IL': 0, 'IN': 1})

            df['policy_csl'] = df['policy_csl'].map({'500/1000': 2, '250/500': 1, '100/300': 0})

            df['insured_sex'] = df['insured_sex'].map({'MALE': 1, 'FEMALE': 0})

            df['insured_education_level'] = df['insured_education_level'].map(
                {'JD': 3, 'High School': 2, 'Associate': 0, 'MD': 4, 'Masters': 5, 'PhD': 6, 'College': 1})

            df['insured_occupation'] = df['insured_occupation'].map(
                {'machine-op-inspct': 6, 'prof-specialty': 9, 'tech-support': 12, 'sales': 11, 'exec-managerial': 3,
                 'craft-repair': 2, 'transport-moving': 13, 'other-service': 8, 'priv-house-serv': 7, 'armed-forces': 1,
                 'adm-clerical': 0, 'protective-serv': 10, 'handlers-cleaners': 5, 'farming-fishing': 4})

            df['insured_hobbies'] = df['insured_hobbies'].map(
                {'sleeping': 17, 'reading': 15, 'board-games': 2, 'bungie-jumping': 3,
                 'base-jumping': 0, 'golf': 9, 'camping': 4, 'dancing': 7, 'skydiving': 16,
                 'movies': 12, 'hiking': 10, 'yachting': 19, 'paintball': 8, 'chess': 5,
                 'kayaking': 11, 'polo': 14, 'basketball': 1, 'video-games': 18, 'cross-fit': 6,
                 'exercise': 13})

            df['insured_relationship'] = df['insured_relationship'].map(
                {'husband': 0, 'other-relative': 2, 'own-child': 3, 'unmarried': 4, 'wife': 5,
                 'not-in-family': 1})

            df['incident_type'] = df['incident_type'].map(
                {'Single Vehicle Collision': 2, 'Vehicle Theft': 3, 'Multi-vehicle Collision': 0, 'Parked Car': 1})

            df['collision_type'] = df['collision_type'].map(
                {'Side Collision': 2, 'Multi-vehicle Collision': 0, 'Rear Collision': 1})

            df['incident_severity'] = df['incident_severity'].map(
                {'Minor Damage': 1, 'Total Loss': 2, 'Major Damage': 0, 'Trivial Damage': 3})

            df['authorities_contacted'] = df['authorities_contacted'].map(
                {'Police': 4, 'Fire': 1, 'Other': 3, 'Ambulance': 0, 'None': 2})

            df['incident_state'] = df['incident_state'].map(
                {'NY': 1, 'SC': 4, 'WV': 6, 'NC': 5, 'VA': 0, 'PA': 3, 'OH': 2})

            df['incident_city'] = df['incident_city'].map(
                {'Springfield': 6, 'Arlington': 0, 'Columbus': 1, 'Northbend': 3, 'Hillsdale': 2, 'Riverwood': 5,
                 'Northbrook': 4})

            df['property_damage'] = df['property_damage'].map({'YES': 1, 'NO': 0})

            df['police_report_available'] = df['police_report_available'].map({'YES': 1, 'NO': 0})

            df['auto_make'] = df['auto_make'].map(
                {'Honda': 6, 'prof-specialty': 9, 'Saab': 11, 'Chevrolet': 3,
                 'Ford': 2, 'Mercedes': 8, 'Jeep': 7, 'Audi': 1,
                 'Accura': 0, 'Dodge': 10, 'BMW': 5, 'Suburu': 4, 'Toyota': 12, 'Volkswagen': 13})

            df['auto_model'] = df['auto_model'].map(
                {'92x': 9, 'E400': 12, 'RAM': 30, 'Tahoe': 34, 'RSX': 31, '95': 3, 'Pathfinder': 29, 'A5': 21,
                 'Camry': 1, 'F150': 14, 'A3': 27, 'Highlander': 18, 'Neon': 4, 'MDX': 23, 'Maxima': 13,
                 'Legacy': 5, 'TL': 11, 'Impreza': 33, 'Forrestor': 15, 'Escape': 26, 'Corolla': 8,
                 '3 Series': 0, 'C300': 7, 'Wrangler': 36, 'M5': 22, 'X5': 37, 'Civic': 32, 'Passat': 28,
                 'Silverado': 10, 'CRV': 24, '93': 2, 'Accord': 6, 'X6': 38, 'Malibu': 25, 'Fusion': 16,
                 'Jetta': 20, 'ML350': 19, 'Ultima': 35, 'Grand Cherokee': 17})

            res = df.to_dict('records')
            return res[0]

        except ValueError:
            return Response("Value Error Occurred! %s" % ValueError)
        except KeyError:
            return Response("Key Error Occurred! %s" % KeyError)
        except Exception as e:
            return Response("Exception Error Occurred! %s" % e)

    def get_feat_importance(self, m):
        return pd.DataFrame({'feature': self.cols, 'imp': m.feature_importances_}
                            ).sort_values('imp', ascending=False)

