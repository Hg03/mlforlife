---
icon: lucide/between-horizontal-end
---

# 2. OOPS & DBMS


## OOPS

### Class

A **class** is a blueprint that defines the structure and behavior of objects. It specifies what data something will hold (fields) and what actions it can perform (methods).

**Real-World Analogy**: Think of it like an architectural blueprint for a house. The blueprint specifies the number of rooms, doors, and windows. But you can’t live in a blueprint. You need to build an actual house from it.

### Objects

An object is a concrete instance of a class. It has actual values for the fields defined in the class.

If the class is a template, each object is a filled-in copy. You can create many objects from the same class, and each one is independent.

```python
class User:
    def __init__(self, name, role):
        self.name = name  # Attribute
        self.role = role  # Attribute

# Creating Objects (Instances)
alice = User("Alice", "Admin")
bob = User("Bob", "Developer")

print(f"{alice.name} is an {alice.role}")
```

Each object has its own copy of the fields. Changing alice‘s role doesn’t affect bob. They’re independent instances built from the same template.

Classes and objects let you group related data and behavior together. But in larger systems, you often need to define what behaviors must exist without specifying how they work. That’s where interfaces come in.

### Interfaces

An interface is a contract. It defines a set of methods that a class must implement, without specifying how they should work.

Think about payment processing in an e-commerce app. You need to charge customers, but you don’t want to be locked into a single payment provider. So, you define a contract that says “any payment gateway must support charging and refunding,” and then Stripe, PayPal, Razorypay or any future provider can plug in.

```python
from abc import ABC, abstractmethod

class PaymentGateway(ABC):
    @abstractmethod
    def process_payment(self, amount):
        pass

class Stripe(PaymentGateway):
    def process_payment(self, amount):
        return f"Processing ${amount} via Stripe."

class PayPal(PaymentGateway):
    def process_payment(self, amount):
        return f"Processing ${amount} via PayPal."
```

### Encapsulation

Encapsulation is the practice of bundling data and methods together in a class while restricting direct access to the internal data. You expose a controlled public interface and hide everything else.

Consider a rate limiter. Other parts of your system only need to ask “can this user make another request?” They shouldn’t be able to directly mess with the internal counters or reset the time window.

```python
class RateLimiter:
    def __init__(self):
        self.__request_count = 0  # Private attribute

    def allow_request(self):
        if self.__request_count < 5:
            self.__request_count += 1
            return True
        return False

limiter = RateLimiter()
# limiter.__request_count  # This would raise an AttributeError
```

### Abstraction

**Abstraction** is about hiding unnecessary complexity and exposing only what the user needs. While encapsulation hides data, abstraction hides implementation details.

**Real-World Analogy**: Think about sending a message through Slack. You type a message and hit send. Behind the scenes, there’s WebSocket management, message serialization, retry logic, delivery confirmation, and push notifications. You don’t deal with any of that. The complexity is abstracted away behind a simple action. 

Abstraction simplifies how you interact with objects. But what if multiple classes share the same data and behavior? That’s where inheritance steps in.

```python
class SlackClient:
    def send_message(self, msg):
        self.__connect_to_websocket()
        self.__serialize(msg)
        print(f"Message sent: {msg}")

    def __connect_to_websocket(self):
        pass # Complex logic hidden here

    def __serialize(self, data):
        pass # Complex logic hidden here

client = SlackClient()
client.send_message("Hello Team!")
```

### Inheritance

Inheritance lets a new class (child) derive from an existing class (parent), inheriting its fields and methods. The child class can reuse the parent’s code, add new behavior, or override existing behavior.

In an event-driven system, every event needs a timestamp, an event ID, and a source. But each specific event type carries its own payload. Instead of duplicating the common fields in every event class, you define them once in a base class.

```python
class Event:
    def __init__(self, timestamp, event_id):
        self.timestamp = timestamp
        self.event_id = event_id

class MessageEvent(Event):
    def __init__(self, timestamp, event_id, content):
        super().__init__(timestamp, event_id) # Call parent constructor
        self.content = content

msg = MessageEvent("10:00 AM", "E123", "Hello!")
```

Inheritance lets classes share structure and behavior. But what happens when you call the same method on different child classes and get different results?

That’s polymorphism.

### Polymorphism

Polymorphism means “many forms.” It allows objects of different types to be treated through a common interface, with each type providing its own behavior.

There are two types:

Compile-time (method overloading): same method name, different parameters

```python
class Calculator:
    # Simulating overloading using default parameters
    def add(self, a, b, c = 0):
        return a + b + c

calc = Calculator()

# Same method name, different number of arguments
print(calc.add(5, 10))      # Output: 15
print(calc.add(5, 10, 20))  # Output: 35
```

Runtime (method overriding): same method signature, different implementations in child classes

```python
class Notification:
    def send(self):
        print("Sending a generic notification")

class Email(Notification):
    def send(self):
        print("Sending an Email with SMTP headers...")

class SMS(Notification):
    def send(self):
        print("Sending an SMS via Telephony API...")

# The specific 'send' method is determined at runtime
notifications = [Email(), SMS(), Notification()]

for app in notifications:
    app.send()
```

## DBMS

### 1) What is DBMS?
A DBMS is software that manages databases, providing an interface to store, retrieve, and manipulate data efficiently and securely.


### 2) What is a Database?
A Database is an organized, consistent, and logical collection of data that can easily be updated, accessed, and managed. Database mostly contains sets of tables or objects which consist of records and fields. 

### 3) Explain the difference between a database and a DBMS?
A database is a collection of related data, while a DBMS is software used to manage, store, and retrieve data efficiently from the database.


### 4) Advantages of DBMS over File Systems?

- **Data Redundancy and Inconsistency**: \
Redundancy means repeating the data in a system. In a normal file system, there is a high chance that there can be various files of the same data used by different users for specific purposes. If any user changes the data in its files, then the changes are not reflected in all files. This creates inconsistency in the data, and it may lead to the failure of the system. But in the DBMS, there is only one repository of data, and multiple users can use it. If any user changes the data, then it is reflected to each user as they are using the same repository.

- **Data Sharing**: \
In the normal file system, data sharing is too difficult because file sharing is a complex task. In DBMS, all the data is centralized, so data sharing is a very easy task.

- **Data Concurrency**: \
When more than one user accesses the database simultaneously, then it is called concurrency. In a file system, when multiple users are using the files at the same time, then there may be a chance of anomalies in the data due to changes, and it does not provide any method to detect anomalies. But in DBMS, we have a locking system to detect the anomalies so we can protect the data.

- **Data Searching**: \
To search the data in a file system, we have to write a specific program and run it. In DBMS, we have query languages by which we can write small queries to get the data we want from the database. We can use various query languages, like MySQL, Oracle, etc., for a database to search and retrieve the data.

- **Data Integrity**: \
When we insert new data into the database, we require some specific constraints on the data like integer or not null, etc. The file system does not provide any system to check the constraints, whereas DBMS has the functionality to check the constraints on the data, and it allows user defined data types.


### 5) What is the different languages present in DBMS?

- **DDL(Data Definition Language)**:  It contains commands which are required to define the database. \ 
E.g., CREATE, ALTER, DROP, TRUNCATE, RENAME, etc.
- **DML(Data Manipulation Language)**: It contains commands which are required to manipulate the data present in the database. \
E.g., SELECT, UPDATE, INSERT, DELETE, etc.
- **DCL(Data Control Language)**:  It contains commands which are required to deal with the user permissions and controls of the database system. \
E.g., GRANT and REVOKE.
- **TCL(Transaction Control Language)**:  It contains commands which are required to deal with the transaction of the database. \
E.g., COMMIT, ROLLBACK, and SAVEPOINT.

### 6) ACID properties in DBMS?

ACID stands for Atomicity, Consistency, Isolation, and Durability in a DBMS these are those properties that ensure a safe and secure way of sharing data among multiple users.

- **Atomicity**: This property reflects the concept of either executing the whole query or executing nothing at all, which implies that if an update occurs in a database then that update should either be reflected in the whole database or should not be reflected at all.
- **Consistency**: This property ensures that the data remains consistent before and after a transaction in a database.
- **Isolation**: This property ensures that each transaction is occurring independently of the others. This implies that the state of an ongoing transaction doesn’t affect the state of another ongoing transaction.
- **Durability**: This property ensures that the data is not lost in cases of a system failure or restart and is present in the same state as it was before the system failure or restart.


### 7) Difference between the DELETE and TRUNCATE command in a DBMS?

- DELETE Command:
    - It removes rows from a table one by one with transaction logging
    - It can be rolled back if required.

- TRUNCATE Command:
    - It removes all rows at once without transaction logging.
    - It can't be rolled back.

### 8) What is meant by Normalization and Denormalization?

**Normalization** is a process of reducing redundancy by organizing the data into multiple tables. Normalization leads to better usage of disk spaces and makes it easier to maintain the integrity of the database.  

**Denormalization** is the reverse process of normalization as it combines the tables which have been normalized into a single table so that data retrieval becomes faster. JOIN operation allows us to create a denormalized form of the data by reversing the normalization. 


### 9) Different types of Normalization forms in a DBMS?

- 1NF: It is known as the first normal form and is the simplest type of normalization that you can implement in a database. A table to be in its first normal form should satisfy the following conditions:
    - Every column must have a single value and should be atomic.
    - Duplicate columns from the same table should be removed.
    - Separate tables should be created for each group of related data and each row should be identified with a unique column.
 
- 2NF: It is known as the second normal form. A table to be in its second normal form should satisfy the following conditions:
    - The table should be in its 1NF i.e. satisfy all the conditions of 1NF.
    - Every non-prime attribute of the table should be fully functionally dependent on the primary key i.e. every non-key attribute should be dependent on the primary key in such a way that if any key element is deleted then even the non_key element will be saved in the database.
 
- 3NF: It is known as the third normal form. A table to be in its third normal form should satisfy the following conditions:
    - The table should be in its 2NF i.e. satisfy all the conditions of 2NF.
    - There is no transitive functional dependency of one attribute on any attribute in the same table.
 
- BCNF: BCNF stands for Boyce-Codd Normal Form and is an advanced form of 3NF. It is also referred to as 3.5NF for the same reason. A table to be in its BCNF normal form should satisfy the following conditions:
    - The table should be in its 3NF i.e. satisfy all the conditions of 3NF.
    - For every functional dependency of any attribute A on B
(A->B), A should be the super key of the table. It simply implies that A can’t be a non-prime attribute if B is a prime attribute.


### 10) What is an Entity-Relationship Diagram (ER-Diagram)?  
An ER-Diagram is a visual representation of the relationships among entities in a database, showing how different tables are connected.


### 11) Different types of keys in a database?

- **Primary key**: The Primary key is an attribute in a table that can uniquely identify each record in a table. It is compulsory for every table.
- **Unique Key**: The unique key is very similar to the primary key except that primary keys don’t allow NULL values in the column but unique keys allow them.
- **Foreign key**: The Foreign key is a primary key from one table, which has a relationship with another table. It acts as a cross-reference between tables.


### 12) What is a lock. Explain the difference between a shared lock and an exclusive lock?

A database lock is a mechanism to protect a shared piece of data from getting updated by two or more database users at the same time. When a single database user or session has acquired a lock then no other database user or session can modify that data until the lock is released.

- **Shared lock**: Shared lock is required for reading a data item. In the shared lock, many transactions may hold a lock on the same data item. When more than one transaction is allowed to read the data items then that is known as the shared lock.

- **Exclusive lock**: When any transaction is about to perform the write operation, then the lock on the data item is an exclusive lock. Because, if we allow more than one transaction then that will lead to the inconsistency in the database.


### 13) What are Views in DBMS?

A view is a virtual table that is derived from one or more base tables or other views. It does not store any data itself but represents a tailored, pre-defined query that simplifies data retrieval. Views act as a layer of abstraction over the underlying tables, providing a more user-friendly and secure way to interact with the data.

**Benefits of using views**:

- **Data Abstraction**: Views allow users to work with a simplified representation of the data, hiding unnecessary details and complexity.
- **Security**: Views can be used to restrict access to certain columns or rows, providing a level of security by only showing specific data to specific users.
- **Simplified Querying**: Complex queries can be encapsulated within views, making it easier for users to retrieve the desired information without writing complex SQL statements.
- **Data Independence**: If the underlying schema changes, views can remain the same, and applications using the views will not be affected.


### 14) What is a Join? List its different types.

The SQL Join clause is used to combine records (rows) from two or more tables in a SQL database based on a related column between the two.

There are four different types of JOINs in SQL:


- **INNER JOIN**: Retrieves records that have matching values in both tables involved in the join. This is the widely used join for queries.
- **LEFT OUTER JOIN**: Retrieves all the records/rows from the left and the matched records/rows from the right table.
- **RIGHT OUTER JOIN**: Retrieves all the records/rows from the right and the matched records/rows from the left table.
- **FULL OUTER JOIN**: Retrieves all the records where there is a match in either the left or right table.


### 15) What is a Self-Join?
A self JOIN is a case of regular join where a table is joined to itself based on some relation between its own column(s). Self-join uses the INNER JOIN or LEFT JOIN clause and a table alias is used to assign different names to the table within the query.


### 16) What is a Cross-Join?
Cross join can be defined as a cartesian product of the two tables included in the join. The table after join contains the same number of rows as in the cross-product of the number of rows in the two tables. If a WHERE clause is used in cross join then the query will work like an INNER JOIN.



### 17) What is an Index? Difference between Clustered and Non-Clustered Index?

A database index is a data structure that provides a quick lookup of data in a column or columns of a table. It enhances the speed of operations accessing data from a database table at the cost of additional writes and memory to maintain the index data structure.

#### Difference between Clustered and Non-Clustered Index

- Clustered index modifies the way records are stored in a database based on the indexed column. A non-clustered index creates a separate entity within the table which references the original table.
- Clustered index is used for easy and speedy retrieval of data from the database, whereas, fetching records from the non-clustered index is relatively slower.
- In SQL, a table can have a single clustered index whereas it can have multiple non-clustered indexes.

---
























