```python

# This is a base file for the Python helper module.
# Import necessary classes and functions from the semt_py package
import semt_py
import getpass
from semt_py import AuthManager
from semt_py.extension_manager import ExtensionManager
from semt_py.reconciliation_manager import ReconciliationManager
from semt_py.utils import Utility
from semt_py.dataset_manager import DatasetManager
from semt_py.table_manager import TableManager
from semt_py.modification_manager import ModificationManager

def get_input_with_default(prompt, default):
    user_input = input(f"{prompt} (default: {default}): ").strip()
    return user_input if user_input else default

base_url = get_input_with_default("Enter base URL or press Enter to keep default", "http://localhost:3003")
username = get_input_with_default("Enter your username", "agazzi.ruben99@gmail.com")
default_password = "Gu_yaU-MvG"
password_prompt = f"Enter your password (default: use stored password): "
password_input = getpass.getpass(password_prompt)
password = password_input if password_input else default_password
api_url = get_input_with_default("Enter API URL or press Enter to keep default", "http://localhost:3003/api")

Auth_manager = AuthManager(api_url, username, password)
token = Auth_manager.get_token()
reconciliation_manager = ReconciliationManager(base_url, Auth_manager)
dataset_manager = DatasetManager(base_url, Auth_manager)
table_manager = TableManager(base_url, Auth_manager)
extension_manager = ExtensionManager(base_url, token)
utility = Utility(base_url, Auth_manager)

```


```python

# Load a dataset into a DataFrame
import pandas as pd

# Note: get_input_with_default is defined in the main file
# Reusing it here for consistency

dataset_id = get_input_with_default("Enter dataset_id or press Enter to keep default", "0")
table_name = get_input_with_default("Enter table_name or press Enter to keep default", "test_table-2025-09-05_13-27")

df = pd.read_csv('./table_1.csv')

# Delete specified columns if any
columns_to_delete = []
if columns_to_delete and columns_to_delete != ['']:
    for col in columns_to_delete:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Deleted column: {col}")
        else:
            print(f"Column '{col}' not found in table")
    print(f"Columns deleted: {[col for col in columns_to_delete if col in df.columns]}")

table_id, message, table_data = table_manager.add_table(dataset_id, df, table_name)

# Extract the table ID
# Alternative method if above doesn't work:
# return_data = dataset_manager.add_table_to_dataset(dataset_id, df, table_name)
# data_dict = return_data[1]  # The dictionary containing table info
# table_id = data_dict['tables'][0]['id']

```

## Operation 1: Reconciliation for column buyer by wikidataAlligator


```python


reconciliator_id = "wikidataAlligator"
optional_columns = []  # Replace with actual optional columns if needed
column_name = "buyer"
try:
    table_data = table_manager.get_table(dataset_id, table_id)
    reconciled_table, backend_payload = reconciliation_manager.reconcile(
        table_data,
        column_name,
        reconciliator_id,
        optional_columns
    )
    payload = backend_payload

    successMessage, sentPayload = utility.push_to_backend(
    dataset_id,
    table_id,
    payload,
    debug=False
    )

    print(successMessage)
    # Display the full reconciled table
    html_table = Utility.display_json_table(reconciled_table)
    # Or display with specific parameters (example)
    html_table = Utility.display_json_table(
        json_table=reconciled_table,
        number_of_rows=4,  # Show 4 rows
        from_row=0,        # Start from first row
    )
except Exception as e:
    print(f"An error occurred during reconciliation: {e}")
    # Handle the exception as needed, e.g., log it or re-raise it

```

## Operation 3: Extension for column buyer by llmClassifier


```python

try:
    table_data = table_manager.get_table(dataset_id, table_id)

    extended_table, extension_payload = extension_manager.extend_column(
        table=table_data,
        column_name="buyer",
        extender_id="llmClassifier",
        properties=[],
        other_params={}
    )
    payload = extension_payload

    successMessage, sentPayload = utility.push_to_backend(
        dataset_id,
        table_id,
        payload,
        debug=False
    )

    print(successMessage)
    # Display the full extended table
    html_table = Utility.display_json_table(extended_table)
    # Or display with specific parameters (example)
    html_table = Utility.display_json_table(
        json_table=extended_table,
        number_of_rows=4,  # Show 4 rows
        from_row=0,        # Start from first row
    )
except Exception as e:
    print(f"An error occurred during extension: {e}")

```

## Operation 4: Extension for column buyer by reconciledColumnExt


```python

try:
    table_data = table_manager.get_table(dataset_id, table_id)

    extended_table, extension_payload = extension_manager.extend_column(
        table=table_data,
        column_name="buyer",
        extender_id="reconciledColumnExt",
        properties=["id"],
        other_params={}
    )
    payload = extension_payload

    successMessage, sentPayload = utility.push_to_backend(
        dataset_id,
        table_id,
        payload,
        debug=False
    )

    print(successMessage)
    # Display the full extended table
    html_table = Utility.display_json_table(extended_table)
    # Or display with specific parameters (example)
    html_table = Utility.display_json_table(
        json_table=extended_table,
        number_of_rows=4,  # Show 4 rows
        from_row=0,        # Start from first row
    )
except Exception as e:
    print(f"An error occurred during extension: {e}")

```
