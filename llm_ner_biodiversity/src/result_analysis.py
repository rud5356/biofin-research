import json

import pandas as pd


def extract_by_type(df: pd.DataFrame, entity_type: str) -> list[str]:
    all_entities: list[str] = []
    for entities_json in df["entities"]:
        try:
            entities = json.loads(entities_json)
        except (TypeError, json.JSONDecodeError):
            continue

        for entity in entities:
            if entity.get("type") == entity_type:
                all_entities.append(entity.get("text", ""))
    return all_entities


def summarize_results(df: pd.DataFrame) -> None:
    if df.empty:
        print("No results were generated.")
        return

    species_list = extract_by_type(df, "SPECIES")
    location_list = extract_by_type(df, "LOCATION")

    print(f"\nExtracted species: {len(species_list)}")
    print(f"Extracted locations: {len(location_list)}")

    if species_list:
        print("Top 10 species:")
        print(pd.Series(species_list).value_counts().head(10))

    print(f"\nParse failure rate: {df['parse_error'].mean():.1%}")
    print(f"Average processing time: {df['elapsed_sec'].mean():.1f} sec")
