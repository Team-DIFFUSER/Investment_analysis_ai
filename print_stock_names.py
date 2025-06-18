from database.database import DatabaseManager

def main():
    db = DatabaseManager()
    with db.get_db_cursor() as cursor:
        cursor.execute('SELECT DISTINCT stock_name FROM stock_prices ORDER BY stock_name;')
        results = cursor.fetchall()
        print('DB에 저장된 종목명:')
        for row in results:
            print('-', row['stock_name'])
    db.close()

if __name__ == '__main__':
    main() 