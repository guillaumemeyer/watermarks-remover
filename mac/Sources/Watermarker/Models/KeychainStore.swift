import Foundation
import Security

/// The OpenRouter API key, kept in the login keychain rather than in the
/// key-value store.
///
/// `kSecAttrSynchronizable` is what carries the item to the user's other Macs:
/// iCloud Keychain is the CloudKit-backed store Apple intends for secrets, and
/// unlike `NSUbiquitousKeyValueStore` it never lands in a readable plist. When
/// the build is not signed with an iCloud-capable identity, the synchronizable
/// write fails and we fall back to a local-only item so the app still works.
enum KeychainStore {
    /// Where the secret ended up, so the UI can be honest about sync.
    enum Placement: Equatable {
        case synchronized
        case localOnly
    }

    enum KeychainError: LocalizedError {
        case unexpectedStatus(OSStatus)

        var errorDescription: String? {
            switch self {
            case .unexpectedStatus(let status):
                let detail = SecCopyErrorMessageString(status, nil) as String?
                return detail ?? "Keychain error \(status)."
            }
        }
    }

    private static func baseQuery(service: String, account: String,
                                  synchronizable: Bool) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecAttrSynchronizable as String: synchronizable ? kCFBooleanTrue! : kCFBooleanFalse!,
        ]
    }

    /// Read the secret, preferring the synced item over a local leftover, and
    /// report which of the two answered.
    static func read(service: String, account: String) -> (value: String, placement: Placement)? {
        for synchronizable in [true, false] {
            var query = baseQuery(service: service, account: account,
                                  synchronizable: synchronizable)
            query[kSecReturnData as String] = kCFBooleanTrue!
            query[kSecMatchLimit as String] = kSecMatchLimitOne
            var item: CFTypeRef?
            let status = SecItemCopyMatching(query as CFDictionary, &item)
            if status == errSecSuccess, let data = item as? Data,
               let value = String(data: data, encoding: .utf8), !value.isEmpty {
                return (value, synchronizable ? .synchronized : .localOnly)
            }
        }
        return nil
    }

    /// Write the secret, trying iCloud Keychain first and reporting where it
    /// actually landed. An empty value deletes the item.
    @discardableResult
    static func write(_ value: String, service: String, account: String) throws -> Placement {
        guard !value.isEmpty else {
            delete(service: service, account: account)
            return .localOnly
        }
        let data = Data(value.utf8)
        var lastStatus: OSStatus = errSecSuccess
        for synchronizable in [true, false] {
            let query = baseQuery(service: service, account: account,
                                  synchronizable: synchronizable)
            let attributes: [String: Any] = [
                kSecValueData as String: data,
                kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlocked,
            ]
            var status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
            if status == errSecItemNotFound {
                var insert = query
                insert.merge(attributes) { _, new in new }
                status = SecItemAdd(insert as CFDictionary, nil)
            }
            if status == errSecSuccess {
                // The item is in one place now; clear the other copy so a stale
                // key can never win the read above.
                deleteOne(service: service, account: account,
                          synchronizable: !synchronizable)
                return synchronizable ? .synchronized : .localOnly
            }
            lastStatus = status
        }
        throw KeychainError.unexpectedStatus(lastStatus)
    }

    static func delete(service: String, account: String) {
        deleteOne(service: service, account: account, synchronizable: true)
        deleteOne(service: service, account: account, synchronizable: false)
    }

    private static func deleteOne(service: String, account: String, synchronizable: Bool) {
        SecItemDelete(baseQuery(service: service, account: account,
                                synchronizable: synchronizable) as CFDictionary)
    }
}
